import copy

import torch
from torch import nn


def vectomodel(vec,net):
    start_idx = 0
    net_glob = copy.deepcopy(net)
    for name, module in net_glob.named_children():
        if isinstance(module, nn.Sequential):
            # 遍历子模块
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module ,nn.BatchNorm2d):
                    # 权重
                    num_elements = sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中                    # 偏置
                    num_elements = sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                    num_elements = sub_module.running_mean.numel()
                    sub_module.running_mean.data = vec[start_idx:start_idx + num_elements].view(
                        sub_module.running_mean.size())
                    start_idx += num_elements
                    num_elements = sub_module.running_var.numel()
                    sub_module.running_var.data = vec[start_idx:start_idx + num_elements].view(
                        sub_module.running_var.size())
                    start_idx += num_elements
                elif isinstance(sub_module, nn.Conv2d):
                    # 权重
                    num_elements = sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                    # 偏置
                    num_elements = sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                elif isinstance(sub_module, nn.Linear):
                    # 权重
                    num_elements = sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                    # 偏置
                    num_elements = sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
    return net_glob

def vectomodel2(vec,net):
    start_idx = 0
    net_glob = copy.deepcopy(net)
    for name, module in net_glob.named_children():
        if isinstance(module, nn.Sequential):
            # 遍历子模块
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module ,nn.BatchNorm2d):
                    # 权重
                    num_elements = sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中                    # 偏置
                    num_elements = sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                    num_elements = sub_module.running_mean.numel()
                    sub_module.running_mean.data = vec[start_idx:start_idx + num_elements].view(
                        sub_module.running_mean.size())
                    start_idx += num_elements
                    num_elements = sub_module.running_var.numel()
                    sub_module.running_var.data = vec[start_idx:start_idx + num_elements].view(
                        sub_module.running_var.size())
                    start_idx += num_elements
                    num_elements = sub_module.num_batches_tracked.numel()
                    sub_module.num_batches_tracked.data = vec[start_idx:start_idx + num_elements].view(
                        sub_module.num_batches_tracked.size())
                    start_idx += num_elements
                elif isinstance(sub_module, nn.Conv2d):
                    # 权重
                    num_elements = sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                    # 偏置
                    num_elements = sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                elif isinstance(sub_module, nn.Linear):
                    # 权重
                    num_elements = sub_module.weight.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.weight.data = vec[start_idx:start_idx + num_elements].view(sub_module.weight.size())
                    # 更新索引
                    start_idx += num_elements
                    # 将参数添加到字典中
                    # 偏置
                    num_elements = sub_module.bias.numel()
                    # 从 output 中提取相应数量的元素
                    sub_module.bias.data = vec[start_idx:start_idx + num_elements].view(sub_module.bias.size())
                    # 更新索引
                    start_idx += num_elements
    return net_glob


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

