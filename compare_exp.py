from merge import MergeLora
from eval import Eval, LoadDataset
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '--base_model',
    type=str,
    help='Path of base model')

args = parser.parse_args()
# lora_0 = '/home/sunyuhan/syh/sunyuhan/zju/Multi-LoRA-LLM/output_dir/roberta-base-mnli-lora-epoch-20'
# lora_1 = '/home/sunyuhan/syh/sunyuhan/zju/Multi-LoRA-LLM/output_dir/roberta-base-rte-lora-epoch-100'
#loras = [lora_0, lora_1]
loras = ['yuuhan/roberta-base-mnli-lora', 'yuuhan/roberta-base-rte-lora']
mnli_dataloader = LoadDataset(args.base_model, task = 'mnli')
rte_loader = LoadDataset(args.base_model, task = 'rte')
for alpha in [0.1 * i for i in range(11)]:
    merged = MergeLora(loras, [alpha, 1-alpha])
    mnli = Eval(mnli_dataloader, merged[0], task = 'mnli')
    rte = Eval(rte_loader, merged[1], task = 'rte')
    print(f'MNLI acc: {mnli["accuracy"]}, RTE acc: {rte["accuracy"]}')