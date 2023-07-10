import pdb
import torch
import os
import shutil
from peft import PeftModel, LoraConfig

def MergeLora(loras:list, coefs:list, dests = ['output_dir/merged_0', 'output_dir/merged_1']):
    lora_dicts = [{k:v.cpu() for k,v in torch.load(f'{lora}/adapter_model.bin').items()} for lora in loras]
    out_dicts = lora_dicts.copy()
    for i in range(len(lora_dicts)):
        for k in lora_dicts[i].keys():
            if not 'classifier' in k:
                out_dicts[i][k] = sum(lora[k] * coef for lora, coef in zip(lora_dicts, coefs))
    for i in range(len(dests)):
        os.makedirs(dests[i], exist_ok=True)
        shutil.copy(f'{loras[i]}/adapter_config.json', dests[i])
        torch.save(out_dicts[i], f'{dests[i]}/adapter_model.bin')
    return dests

def MergeLorasToBaseModel(base_model, loras:list, coefs:list, use_head_list:list):
    tmpdir = 'tmpdir'
    # pdb.set_trace()
    os.makedirs(tmpdir, exist_ok=True)
    for lora, coef, use_head in zip(loras, coefs, use_head_list):
        print(lora, coef, use_head)
        shutil.copy(f'{lora}/adapter_config.json', tmpdir)
        state_dict = {k:(coef*v.cpu()) for k,v in torch.load(f'{lora}/adapter_model.bin').items()}
        if not use_head:
            keys = list(state_dict.keys())
            for k in keys:
                if 'out_proj' in k:
                    state_dict.pop(k)
            torch.save(state_dict, f'{tmpdir}/adapter_model.bin')
            pdb.set_trace()
            base_model.base_model.set_adapter(tmpdir)
        else:
            base_model = PeftModel.from_pretrained(base_model, tmpdir)
            base_model.base_model.merge_and_unload()
    # pdb.set_trace()
    return base_model

def MergeLayerLora(loras:list, coefs:list, dests = ['output_dir/merged_0', 'output_dir/merged_1']):
    lora_dicts = [{k:v.cpu() for k,v in torch.load(f'{lora}/adapter_model.bin').items()} for lora in loras]
    out_dicts = lora_dicts.copy()
    layers = {}
    for lora_dict, coef in zip(lora_dicts, coefs):
        for k, v in lora_dict.items():
            if not 'classifier' in k:
                layers[k] = v * coef
    for out_dict in out_dicts:
        for k in layers.keys():
            out_dict[k] = layers[k]
    # pdb.set_trace()
    for i in range(len(dests)):
        os.makedirs(dests[i], exist_ok=True)
        # shutil.copy(f'{loras[i]}/adapter_config.json', dests[i])
        config = LoraConfig.from_pretrained(loras[i])
        config.layers_to_transform = None
        config.save_pretrained(dests[i])
        torch.save(out_dicts[i], f'{dests[i]}/adapter_model.bin')
    return dests