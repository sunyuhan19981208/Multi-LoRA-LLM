import pdb
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
import t5_encoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import argparse
from moe import MoE
import os
from sentence_transformers import SentenceTransformer
from eval import Eval
# os.environ['CUDA_VISIBLE_DEVICES']='7'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--task',
    default='mnli',
    type=str,
    help='Which task will be choosed in GLUE')

parser.add_argument(
    '--base_model',
    default='/home/sunyuhan/syh/sunyuhan/zju/t5-base',
    type=str,
    help='Base model to be trained')

parser.add_argument(
    '--epoch',
    default=100,
    type=int,
    help='Epoch of training')

args = parser.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
batch_size = 24
model_name_or_path = args.base_model
task = args.task
peft_type = PeftType.LORA
num_epochs = args.epoch
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
embedding_model = SentenceTransformer('/home/sunyuhan/syh/sunyuhan/exp/Multi-LoRA-LLM/scripts/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["premise" if task == 'mnli' else 'sentence1'], examples["hypothesis" if task == 'mnli' else 'sentence2'], truncation=True, max_length=None)
    texts = [examples["premise" if task == 'mnli' else 'sentence1'], examples["hypothesis" if task == 'mnli' else 'sentence2']]
    texts = [f'{x} {y}' for x,y in zip(texts[0], texts[1])]
    embeddings = [embedding_model.encode(x) for x in texts]
    outputs['embeddings'] = embeddings
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "premise" if task == 'mnli' else 'sentence1', "hypothesis"  if task == 'mnli' else 'sentence2'],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)



eval_dataloader = DataLoader(
    tokenized_datasets["validation_matched" if task=='mnli' else 'validation'], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
cls_head = None
def load_expert(lora_model: str) -> PeftModel:
    global model_name_or_path, cls_head
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_labels)
    model.classifier.weight.data = cls_head['base_model.model.classifier.weight']
    model.classifier.bias.data = cls_head['base_model.model.classifier.bias']
    model = PeftModel.from_pretrained(model, lora_model)
    return model

lora_models = ['/home/sunyuhan/syh/sunyuhan/exp/loras/mrpc_lora_t5-base_202307192148',
                '/home/sunyuhan/syh/sunyuhan/exp/loras/rte_lora_t5-base_202307192132']
# experts = [load_expert(lora_model) for lora_model in lora_models]
experts = []


for lora_model in lora_models:
    if task in lora_model:
        cls_head = {k:v.cpu() for k,v in torch.load(os.path.join(lora_model, 'cls_head.bin')).items()}

for lora_model in lora_models:
    experts.append(load_expert(lora_model))
for expert in experts:
    state_dict = get_peft_model_state_dict(expert)
    for k in cls_head.keys():
        state_dict[k] = cls_head[k].cpu()
# pdb.set_trace()
    

model = MoE(input_size=384, output_size=2, num_experts=2, hidden_size=256, noisy_gating=True, k=1, experts=experts)

optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

criterion = nn.CrossEntropyLoss()

# print('Eval')

# res = Eval(eval_dataloader, base_model=experts[0], task='rte')
# print(res)
# res = Eval(eval_dataloader, base_model=experts[1], task='rte')
# print(res)
# exit(0)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)
        outputs, aux_loss = outputs
        loss = criterion(outputs, batch['labels'])
        total_loss = loss + aux_loss
        accelerator.backward(total_loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs, aux_loss = model(**batch)
        predictions = outputs.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    accelerator.print(f"epoch {epoch}:", eval_metric)