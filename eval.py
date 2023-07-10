import torch
from torch.utils.data import DataLoader
from peft import PeftModel

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

import argparse

def LoadDataset(base_model:str, task:str='mnli'):
    batch_size = 32
    model_name_or_path = base_model
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    datasets = load_dataset("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["premise" if task == 'mnli' else 'sentence1'], examples["hypothesis" if task == 'mnli' else 'sentence2'], truncation=True, max_length=None)
        return outputs
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "premise" if task == 'mnli' else 'sentence1', "hypothesis"  if task == 'mnli' else 'sentence2'],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    # Instantiate dataloaders.
    eval_dataloader = DataLoader(
        tokenized_datasets["validation_matched" if task.startswith("mnli") else 'validation'], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return eval_dataloader

def Eval(eval_dataloader:DataLoader, lora_model:str = None, base_model = "/home/sunyuhan/syh/sunyuhan/zju/roberta-base", task:str = "mnli"):
    if isinstance(base_model, str):
        num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
        model = AutoModelForSequenceClassification.from_pretrained(base_model, return_dict=True, num_labels=num_labels)
    else:
        model = base_model
    if lora_model is not None:
        model = PeftModel.from_pretrained(model, lora_model)
    device = 'cuda'
    model = model.to(device)
    metric = evaluate.load("glue", task)
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    return eval_metric

if __name__ == "__main__":
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
        help='Base model')

    parser.add_argument(
        '--lora_model',
        type=str,
        help='Lora model')
    args = parser.parse_args()
    res = Eval(LoadDataset(task = args.task, base_model=args.base_model), base_model=args.base_model, lora_model=args.lora_model, task = args.task)
    print(res)