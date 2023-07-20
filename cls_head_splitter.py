import pdb
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--lora_model',
    default='mnli',
    type=str,
    help='lora model path to be split')
args = parser.parse_args()
path = args.lora_model
total_weights = torch.load(f'{path}/adapter_model.bin')
head_weights = {}
total_keys = list(total_weights.keys())
for k in total_keys:
    if 'classifier' in k:
        head_weights[k] = total_weights.pop(k)
torch.save(total_weights, f'{path}/adapter_model.bin')
torch.save(head_weights, f'{path}/cls_head.bin')
# pdb.set_trace()