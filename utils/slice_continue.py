import copy

import torch
from torch import nn

def vectoslice(M, slices):
    res = [[] for _ in range(len(M))]
    for i, vec in enumerate(M):
        tmp = len(vec)//(slices)
        for j in range(slices - 1):
            res[i].append(vec[j*tmp:(j+1)*tmp])
        res[i].append(vec[(slices - 1)*tmp:])
    return res
def slicetovec(args, slices):
    vec = torch.tensor([]).to(args.device)
    for slice in slices:
        # 将layer拼成一个向量
        vec = torch.cat([vec, slice.view(-1)])
    return vec

def vectomodel_squeezenet(vec, net_glob):
    # 创建字典
    param_dict = {}
    start_idx = 0
    for name, module in net_glob.named_children():
        for sub_name, sub_module in module.named_children():
            # print(f'{name}.{sub_name}')
            for sub_sub_name, sub_sub_module in sub_module.named_children():
                # 遍历权重和偏置
                if isinstance(sub_sub_module, nn.Conv2d):
                    num_elements = sub_sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 偏置
                    num_elements = sub_sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
                elif isinstance(sub_sub_module, nn.Linear):
                    num_elements = sub_sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 偏置
                    num_elements = sub_sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
                for sub_sub_sub_name, sub_sub_sub_module in sub_sub_module.named_children():
                    # 遍历权重和偏置
                    if isinstance(sub_sub_sub_module, nn.Conv2d):
                        num_elements = sub_sub_sub_module.weight.numel()
                        # 从 output 中提取相应数量的元素
                        sub_sub_sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(
                            sub_sub_sub_module.weight.size())
                        # 更新索引
                        start_idx += num_elements
                        # 偏置
                        num_elements = sub_sub_sub_module.bias.numel()
                        # 从 output 中提取相应数量的元素
                        sub_sub_sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(
                            sub_sub_sub_module.bias.size())
                        # 更新索引
                        start_idx += num_elements
    return net_glob