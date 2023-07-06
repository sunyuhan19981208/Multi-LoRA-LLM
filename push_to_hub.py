from peft import PeftModel
from transformers import AutoModelForSequenceClassification
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '--task',
    default='mnli',
    type=str,
    help='Which task will be choosed in GLUE')

parser.add_argument(
    '--base_model',
    type=str,
    help='Base model')

parser.add_argument(
    '--lora_model',
    type=str,
    help='Lora model')

parser.add_argument(
    '--hub_name',
    type=str,
    help='Hub repo name')

args = parser.parse_args()
num_labels = 3 if args.task.startswith("mnli") else 1 if args.task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(args.base_model, return_dict=True, num_labels=num_labels)
model = PeftModel.from_pretrained(model, args.lora_model)
model.push_to_hub(args.hub_name)