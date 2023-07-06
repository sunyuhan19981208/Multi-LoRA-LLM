from transformers import AutoModelForSequenceClassification
from eval import Eval, LoadDataset
from merge import MergeLorasToBaseModel
import argparse
if __name__ == "__main__":
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
        '--lora_models',
        type=str,
        help='Lora model')
    
    parser.add_argument(
        '--coefs',
        type=str,
        help='Coefficients while merging')
    
    parser.add_argument(
        '--use_head',
        type=int,
        help='0 means use the head of first lora and 1 means second...')
    
    args = parser.parse_args()
    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, return_dict=True, num_labels=num_labels)
    loras = args.lora_models.split(',')
    coefs = [float(x) for x in args.coefs.split(',')]
    use_head = [i == args.use_head for i in range(len(loras))]
    model = MergeLorasToBaseModel(model, loras, coefs, use_head)
    res = Eval(LoadDataset(args.base_model, task = args.task), base_model=model, task = args.task)
    print(res)