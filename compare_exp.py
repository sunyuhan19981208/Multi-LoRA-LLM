from merge import MergeLora, MergeLayerLora
from eval import Eval, LoadDataset
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '--base_model',
    type=str,
    default='/home/sunyuhan/syh/sunyuhan/zju/roberta-base',
    help='Path of base model')

parser.add_argument(
    '--layer',
    action="store_true",
    help='Use layer LoRA merge')

args = parser.parse_args()
# loras = ['yuuhan/roberta-base-mnli-lora', 'yuuhan/roberta-base-rte-lora']
mnli_dataloader = LoadDataset(args.base_model, task = 'mnli')
rte_loader = LoadDataset(args.base_model, task = 'rte')

lora_0 = '/home/sunyuhan/syh/sunyuhan/zju/Multi-LoRA-LLM/loras/roberta-base-mnli-lora'
lora_1 = '/home/sunyuhan/syh/sunyuhan/zju/Multi-LoRA-LLM/loras/roberta-base-rte-lora'
loras = [lora_0, lora_1]
for alpha in [0.1 * i for i in range(11)]:
    merged = MergeLora(loras, [alpha, 1-alpha]) if not args.layer else MergeLayerLora(loras, [alpha, 1-alpha]) 
    mnli = Eval(mnli_dataloader, merged[0], task = 'mnli')
    rte = Eval(rte_loader, merged[1], task = 'rte')
    print(f'ALPHA: {alpha}, MNLI acc: {mnli["accuracy"]}, RTE acc: {rte["accuracy"]}')

# Layer Merge
# lora_0 = '/home/sunyuhan/syh/sunyuhan/zju/Multi-LoRA-LLM/loras/roberta-base-mnli-lora-layer0-5'
# lora_1 = '/home/sunyuhan/syh/sunyuhan/zju/Multi-LoRA-LLM/loras/roberta-base-rte-lora-layer6-11'
# loras = [lora_0, lora_1]
# merged = MergeLora(loras, [1.0, 1.0]) if not args.layer else MergeLayerLora(loras, [1.0, 0.5]) 
# mnli = Eval(mnli_dataloader, merged[0], task = 'mnli')
# rte = Eval(rte_loader, merged[1], task = 'rte')
# print(f'MNLI acc: {mnli["accuracy"]}, RTE acc: {rte["accuracy"]}')