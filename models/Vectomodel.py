import copy

import torch
from torch import nn



def vectomodel_vit(vec, net_glob):
    start_idx = 0
    for name, module in net_glob.classifier.named_modules():
        # if 'base_layer' in name:
        #     num_elements = module.weight.numel()
        #     module.weight.data = vec[start_idx:start_idx + num_elements].view(module.weight.size())
        #     start_idx += num_elements
        #     num_elements = module.bias.numel()
        #     module.bias.data = vec[start_idx:start_idx + num_elements].view(module.bias.size())
        #     start_idx += num_elements

        if 'lora_A.default' in name:
            num_elements = module.weight.numel()
            module.weight.data = vec[start_idx:start_idx + num_elements].view(module.weight.size())
            start_idx += num_elements
        if 'lora_B.default' in name:
            num_elements = module.weight.numel()
            module.weight.data = vec[start_idx:start_idx + num_elements].view(module.weight.size())
            start_idx += num_elements
    return net_glob

