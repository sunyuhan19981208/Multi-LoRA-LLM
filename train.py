from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--task',
    default='mnli',
    type=str,
    help='Which task will be choosed in GLUE')

parser.add_argument(
    '--base_model',
    default='/home/sunyuhan/syh/sunyuhan/zju/roberta-base',
    type=str,
    help='Base model to be trained')

parser.add_argument(
    '--layer',
    default=None,
    type=str,
    help='Layers to be trained')

args = parser.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
batch_size = 32
model_name_or_path = args.base_model
task = args.task
peft_type = PeftType.LORA
num_epochs = 100
layer = None if args.layer == None else [int(x) for x in args.layer.split(',')]
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, layers_to_transform=layer)
lr = 3e-4
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["premise" if task == 'mnli' else 'sentence1'], examples["hypothesis" if task == 'mnli' else 'sentence2'], truncation=True, max_length=None)
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
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_labels)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    accelerator.print(f"epoch {epoch}:", eval_metric)
    # accelerator.save_state(f"output_dir/roberta-base-{task}-lora-epoch-{epoch+1}")
    import datetime
    current_time = datetime.datetime.now()
    output_dir = f"output_dir/{task}-{current_time.strftime('%Y-%m-%d-%H-%M')}-epoch-{epoch+1}"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )