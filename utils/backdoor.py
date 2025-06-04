import pickle

import numpy as np
import art
import torchvision
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import load_mnist, preprocess, to_categorical, load_cifar10
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from models.Test import test_img
import os
import struct

from models.Update import DatasetSplit
#生成minist后门攻击样本
from models.add_pattern import add_pattern_bd2, add_pattern_bd3


class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        # 元组中的第一个元素是图像，第二个元素是标签
        self.data = dataset
        # 如果有传入变换函数，就保存下来
        self.transform = transform

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    def __getitem__(self, index):
        # 根据索引返回对应的图像和标签
        image, label = self.data[index]
        # 如果有变换函数，就对图像进行变换
        if self.transform:
            image = self.transform(image)
        # 返回图像和标签
        return image, label




class BackdoorDataset(Dataset):
    def __init__(self, x, y, backdoor):
        self.x = x
        self.y = y
        self.backdoor = backdoor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.backdoor(self.x[index]), self.y[index]

def generate_Mnist_backdoor(num_parties = 3, scale = 1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)
    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties# 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)# 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party] # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]# 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])# 后门攻击标签
    percent_poison = 0.8 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party)) # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)] # 所有标签为正常标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices)) # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices)) # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False) # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1) # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)
    poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)

    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    clean_dataset_train = torch.utils.data.random_split(x_train_parties,
                                                        [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    poisoned_labels_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    return poisoned_dataset_train,clean_dataset_train
def generate_Mnist_backdoor2(num_parties=5, scale=1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)

    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 0.5 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))




    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)

    transforms_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围还原为灰度图像的范围
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)

    # 将图片转为二值图
    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0], cmap='gray')
    # 保存图片为eps格式
    # plt.savefig('./save/mnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    # 保存原图，不加xy轴
    # plt.axis('off')
    # plt.savefig('./save/mnist.png', format='png', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    data_train = []
    data_test = []
    data_clean = []
    # 打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将fmnist的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_train.append((s, i))
    poisoned_dataset_train = MyDataset(data_train, transform=transforms_train)

    plt.figure()
    plt.imshow(poisoned_dataset_train.data[1][0], cmap='gray')
    # 保存图片为eps格式
    # plt.savefig('./save/atk_mnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()


    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)

    # x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_clean.append((s, i))
    clean_dataset = MyDataset(data_clean, transform=transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])

    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_test.append((s, i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform=transforms_test)
    return poisoned_dataset_train, clean_dataset_train, poisoned_test
def generate_Mnist_backdoor_dot(num_parties=2, scale=1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)

    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_single_bd)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 1 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    # remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为正常标签的样本索引

    # target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)
    poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)

    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    clean_dataset_train = torch.utils.data.random_split(x_train_parties,
                                                        [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    return poisoned_dataset_train, clean_dataset_train, poisoned_test
def generate_Mnist_backdoor3(num_parties=2, scale=1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)

    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 1 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    # remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为正常标签的样本索引

    # target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)
    poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)

    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    clean_dataset_train = torch.utils.data.random_split(x_train_parties,
                                                        [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    return poisoned_dataset_train, clean_dataset_train, poisoned_test
def generate_Mnist_backdoorb0_8(num_parties=5, scale=1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)

    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 1 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))




    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)

    transforms_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围还原为灰度图像的范围
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)

    # 将图片转为二值图
    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0], cmap='gray')
    # 保存图片为eps格式
    # plt.savefig('./save/mnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    # 保存原图，不加xy轴
    # plt.axis('off')
    # plt.savefig('./save/mnist.png', format='png', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    data_train = []
    data_test = []
    data_clean = []
    # 打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将fmnist的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_train.append((s, i))
    poisoned_dataset_train = MyDataset(data_train, transform=transforms_train)

    plt.figure()
    plt.imshow(poisoned_dataset_train.data[1][0], cmap='gray')
    # 保存图片为eps格式
    # plt.savefig('./save/atk_mnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()


    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)

    # x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_clean.append((s, i))
    clean_dataset = MyDataset(data_clean, transform=transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])

    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())

    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_test.append((s, i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform=transforms_test)
    return poisoned_dataset_train, clean_dataset_train, poisoned_test

def load_fmnist(path, kind='train'):
    # path是数据集的存放路径，kind是训练集还是测试集
    # 根据kind参数拼接图像和标签文件的路径
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    # 以二进制模式打开图像和标签文件，并读取数据
    with open(images_path, 'rb') as img_file, open(labels_path, 'rb') as lbl_file:
        # 读取图像文件的头部信息，包括魔数，图像数量，行数，列数
        magic, num, rows, cols = struct.unpack('>IIII', img_file.read(16))
        # 读取所有图像的像素值，每个像素值占一个字节
        images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(num, rows, cols)
        # 读取标签文件的头部信息，包括魔数，标签数量
        magic, num = struct.unpack('>II', lbl_file.read(8))
        # 读取所有标签的值，每个标签值占一个字节
        labels = np.frombuffer(lbl_file.read(), dtype=np.uint8)
    # 转换图像的数据类型为浮点数，方便计算
    images = images.astype(np.float32)
    # 获取图像的最大值和最小值
    min_ = np.min(images)
    max_ = np.max(images)
    # 返回图像和标签的数组，以及最大值和最小值
    return images, labels, min_, max_

def generate_FMnist_backdoor(num_parties = 2, scale= 1):
# 加载数据
    path = '../data/fmnist/FashionMNIST/raw'
    x_raw, y_raw, min_, max_ = load_fmnist(path)  # 修改为load_fmnist
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 0.5  # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    # 选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)

    transforms_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.2860,) ,(0.3205,)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,) ,(0.3205,)),
    ])

    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围还原为灰度图像的范围
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)

    #将图片转为二值图
    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0],cmap='gray')
    # 保存图片为eps格式
    plt.savefig('./save/fmnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    data_train = []
    data_test = []
    data_clean = []
    # 打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将fmnist的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_train.append((s, i))
    poisoned_dataset_train = MyDataset(data_train, transform=transforms_train)


    plt.figure()
    plt.imshow(poisoned_dataset_train.data[1][0],cmap='gray')
    # 保存图片为eps格式
    plt.savefig('./save/atk_fmnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()



    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)


    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_clean.append((s, i))
    clean_dataset = MyDataset(data_clean, transform=transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])


    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())


    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_test.append((s, i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform=transforms_test)




    return poisoned_dataset_train, clean_dataset_train, poisoned_test

def generate_FMnist_backdoor2(num_parties = 2, scale= 1):
# 加载数据
    path = '../data/fmnist/FashionMNIST/raw'
    x_raw, y_raw, min_, max_ = load_fmnist(path)  # 修改为load_fmnist
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd2)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 0.5  # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    # 选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)

    transforms_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.2860,) ,(0.3205,)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,) ,(0.3205,)),
    ])

    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围还原为灰度图像的范围
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)

    #将图片转为二值图
    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0],cmap='gray')
    # 保存图片为eps格式
    plt.savefig('./save/fmnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    data_train = []
    data_test = []
    data_clean = []
    # 打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将fmnist的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_train.append((s, i))
    poisoned_dataset_train = MyDataset(data_train, transform=transforms_train)


    plt.figure()
    plt.imshow(poisoned_dataset_train.data[1][0],cmap='gray')
    # 保存图片为eps格式
    plt.savefig('./save/atk_fmnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()


    plt.figure()
    plt.imshow(poisoned_dataset_train.data[2][0], cmap='gray')
    # 保存图片为eps格式
    plt.savefig('./save/atk_fmnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()


    plt.figure()
    plt.imshow(poisoned_dataset_train.data[3][0], cmap='gray')
    # 保存图片为eps格式
    plt.savefig('./save/atk_fmnist.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)


    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_clean.append((s, i))
    clean_dataset = MyDataset(data_clean, transform=transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])


    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())


    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_test.append((s, i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform=transforms_test)




    return poisoned_dataset_train, clean_dataset_train, poisoned_test
def generate_FMnist_backdoor3(num_parties = 2, scale= 1):
# 加载数据
    path = '../data/fmnist/FashionMNIST/raw'
    x_raw, y_raw, min_, max_ = load_fmnist(path)  # 修改为load_fmnist
    # 预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(60000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((60000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本
    backdoor = PoisoningAttackBackdoor(add_pattern_bd3)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 0.8  # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    # 选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    posioned_data_ch = np.expand_dims(poisoned_data, axis=1)
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)

    transforms_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.2860,) ,(0.3205,)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,) ,(0.3205,)),
    ])

    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围还原为灰度图像的范围
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)


    # for i in range(20):
    #     plt.figure()
    #     plt.imshow(poison_before.data[i][0], cmap='gray')
    #     plt.savefig('./save/fmnist' + str(i) + '.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    #     plt.savefig('./save/fmnist' + str(i) + '.png', format='png', bbox_inches='tight', pad_inches=0.0)
    #     plt.show()

    data_train = []
    data_test = []
    data_clean = []
    # 打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将fmnist的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_train.append((s, i))
    poisoned_dataset_train = MyDataset(data_train, transform=transforms_train)



    #保存poisoned_dataset_train的前十张图片
    # for i in range(20):
    #     plt.figure()
    #     plt.imshow(poisoned_dataset_train.data[i][0], cmap='gray')
    #     plt.savefig('./save/atk_fmnist'+str(i)+'.eps', format='eps', bbox_inches='tight', pad_inches=0.0)
    #     plt.savefig('./save/atk_fmnist'+str(i)+'.png', format='png', bbox_inches='tight', pad_inches=0.0)
    #     plt.show()

    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
    print(x_train_parties_ch.shape)
    print(y_train_parties_c.shape)


    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_clean.append((s, i))
    clean_dataset = MyDataset(data_clean, transform=transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])


    x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
    #                                                     [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())


    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将张量形状改为(28,28)
        s = np.squeeze(s)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage(mode='L')(s)
        data_test.append((s, i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform=transforms_test)




    return poisoned_dataset_train, clean_dataset_train, poisoned_test

def generate_Cifar10_backdoor(num_parties=2, scale=1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)



    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(50000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((50000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本

    backdoor = PoisoningAttackBackdoor(insert_image)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 0.5 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 将后门攻击样本的标签转换为PIL image格式
    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage()(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)

    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0])
    #保存图片为eps格式
    plt.savefig('./save/cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()




    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    # posioned_data_ch = np.transpose(poisoned_data, (0, 3, 1, 2))   # 转成通道在前的格式
    posioned_data_ch = poisoned_data
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    #转成通道在前的格式
    # poisoned_x_train_ch = np.transpose(poisoned_x_train, (0, 3, 1, 2))
    poisoned_x_train_ch = poisoned_x_train
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)
    data_train = []
    data_test= []
    data_clean = []

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])



    #打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage()(s)
        data_train.append((s,i))
    poisoned_dataset_train = MyDataset(data_train,transform = transforms_train)
    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    # x_train_parties_ch = np.transpose(x_train_parties, (0, 3, 1, 2))# 转成通道在前的格式
    x_train_parties_ch = x_train_parties
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)


    # x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())

    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
                                                        # [num_samples_per_party for _ in range(1, num_parties)])
    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        s = transforms.ToPILImage()(s)
        data_clean.append((s,i))
    clean_dataset = MyDataset(data_clean, transform = transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())
    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        s = transforms.ToPILImage()(s)
        data_test.append((s,i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform = transforms_test)
    # print(poisoned_dataset_train[0][0].shape,clean_dataset_train[0][0][0].shape, poisoned_test[0][0].shape)
    # print(len(poisoned_dataset_train), len(clean_dataset_train[0]), len(poisoned_test))

    # 显示图片
    plt.figure()
    plt.imshow(poisoned_dataset_train.data[1][0])
    #保存图片为eps格式
    plt.savefig('./save/atk_cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()

    return poisoned_dataset_train, clean_dataset_train, poisoned_test
def generate_Cifar10_backdoor2(num_parties=2, scale=1):
    # 加载数据
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    x_test, y_test = preprocess(x_raw_test, y_raw_test)



    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(50000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((50000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本

    backdoor = PoisoningAttackBackdoor(insert_image)
    example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 后门攻击标签
    percent_poison = 0.5 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 将后门攻击样本的标签转换为PIL image格式
    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage()(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)

    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0])
    #保存图片为eps格式
    plt.savefig('./save/cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()




    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    # posioned_data_ch = np.transpose(poisoned_data, (0, 3, 1, 2))   # 转成通道在前的格式
    posioned_data_ch = poisoned_data
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    #转成通道在前的格式
    # poisoned_x_train_ch = np.transpose(poisoned_x_train, (0, 3, 1, 2))
    poisoned_x_train_ch = poisoned_x_train
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)
    data_train = []
    data_test= []
    data_clean = []

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])



    #打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage()(s)
        data_train.append((s,i))
    poisoned_dataset_train = MyDataset(data_train,transform = transforms_train)
    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    # x_train_parties_ch = np.transpose(x_train_parties, (0, 3, 1, 2))# 转成通道在前的格式
    x_train_parties_ch = x_train_parties
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)


    # x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())

    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
                                                        # [num_samples_per_party for _ in range(1, num_parties)])
    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        s = transforms.ToPILImage()(s)
        data_clean.append((s,i))
    clean_dataset = MyDataset(data_clean, transform = transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())
    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        s = transforms.ToPILImage()(s)
        data_test.append((s,i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform = transforms_test)
    # print(poisoned_dataset_train[0][0].shape,clean_dataset_train[0][0][0].shape, poisoned_test[0][0].shape)
    # print(len(poisoned_dataset_train), len(clean_dataset_train[0]), len(poisoned_test))

    # 显示图片
    for i in range(20):
        plt.figure()
        plt.imshow(poisoned_dataset_train.data[i][0])
        #保存图片为eps格式
        # plt.savefig('./save/atk_cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
        plt.show()
        plt.close()

    return poisoned_dataset_train, clean_dataset_train, poisoned_test




def load_cifar100(data_dir, kind='train'):

    file_path = os.path.join(data_dir, f'{kind}')

    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
        images = data[b'data']
        labels = data[b'fine_labels']

    images = images.reshape(-1, 3, 32, 32)
    images = images.astype('float32') / 255.0  # 归一化到 [0, 1] 范围内
    labels = np.array(labels)

    return images, labels


def generate_Cifar100_backdoor(num_parties=2, scale=1):
    # 加载数据
    path = "../data/cifar100/cifar-100-python"
    x_raw, y_raw= load_cifar100(path,kind='train')
    x_raw_test, y_raw_test = load_cifar100(path,kind='test')
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw, nb_classes=100)
    x_test, y_test = preprocess(x_raw_test, y_raw_test, nb_classes=100)



    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    x_train = x_train.transpose(0, 2, 3, 1)
    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(50000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((50000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签
    # 选择后门攻击样本

    backdoor = PoisoningAttackBackdoor(insert_image)
    example_target = np.array([0] * 99 + [1])  # 后门攻击标签
    percent_poison = 0.5 # 后门攻击样本比例

    # 选择要攻击的样本
    all_indices = np.arange(len(x_train_party))  # 所有样本的索引
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]  # 所有标签为正常标签的样本索引
    #选择标签为0的样本
    # target_indices = all_indices[np.all(y_train_party == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=1)]  # 所有标签为0标签的样本索引

    target_indices = list(set(all_indices) - set(remove_indices))  # 所有标签为后门攻击标签的样本索引
    num_poison = int(percent_poison * len(target_indices))  # 后门攻击样本数量
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)  # 随机选择后门攻击样本

    # 将后门攻击样本的标签转换为PIL image格式
    poison_visual = []
    for s, i in zip(x_train_party, y_train_party):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将s从np转成PIL image图像格式

        s = transforms.ToPILImage()(s)
        poison_visual.append((s, i))
    poison_before = MyDataset(poison_visual)
    if not os.path.exists('./save'):
        os.makedirs('./save')
    # 显示图片
    plt.figure()
    plt.imshow(poison_before.data[1][0])
    #保存图片为eps格式
    plt.savefig('./save/cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()




    # 生成后门攻击样本
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)
    # posioned_data_ch = np.transpose(poisoned_data, (0, 3, 1, 2))   # 转成通道在前的格式
    posioned_data_ch = poisoned_data
    poisoned_labels_c = np.argmax(poisoned_labels, axis=1).astype(int)
    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)  # 标签转换为one-hot编码
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # 生成其他客户端的训练集
    #转成通道在前的格式
    # poisoned_x_train_ch = np.transpose(poisoned_x_train, (0, 3, 1, 2))
    poisoned_x_train_ch = poisoned_x_train
    print('poisoned_x_train_ch.shape:', poisoned_x_train_ch.shape)
    print('poisoned_y_train.shape:', poisoned_y_train.shape)
    data_train = []
    data_test= []
    data_clean = []

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),  # R,G,B每层的归一化用到的均值和方差
    ])



    #打包数据格式
    for s, i in zip(poisoned_x_train_ch, poisoned_y_train):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        # 将s从np转成PIL image图像格式
        s = transforms.ToPILImage()(s)
        data_train.append((s,i))
    poisoned_dataset_train = MyDataset(data_train,transform = transforms_train)
    # poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long())

    num_samples = (num_parties - 1) * num_samples_per_party
    x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    # x_train_parties_ch = np.transpose(x_train_parties, (0, 3, 1, 2))# 转成通道在前的格式
    x_train_parties_ch = x_train_parties
    y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party + num_samples]
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)


    # x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())

    # clean_dataset_train = torch.utils.data.random_split(x_train_parties,
                                                        # [num_samples_per_party for _ in range(1, num_parties)])
    for s, i in zip(x_train_parties_ch, y_train_parties_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        s = transforms.ToPILImage()(s)
        data_clean.append((s,i))
    clean_dataset = MyDataset(data_clean, transform = transforms_train)
    # 将clean_dataset分成num_parties-1个数据集
    clean_dataset_train = torch.utils.data.random_split(clean_dataset,
                                                        [num_samples_per_party for _ in range(1, num_parties)])
    # 全部是后门攻击的图片数据
    # poisoned_test = TensorDataset(torch.Tensor(posioned_data_ch), torch.Tensor(poisoned_labels_c).long())
    for s, i in zip(posioned_data_ch, poisoned_labels_c):
        # 将s的数据范围归一化到[0, 255]
        s = (s - s.min()) / (s.max() - s.min()) * 255
        # 将s的数据类型转换为uint8
        s = s.astype(np.uint8)
        s = transforms.ToPILImage()(s)
        data_test.append((s,i))
    # 转成数据集格式
    poisoned_test = MyDataset(data_test, transform = transforms_test)
    # print(poisoned_dataset_train[0][0].shape,clean_dataset_train[0][0][0].shape, poisoned_test[0][0].shape)
    # print(len(poisoned_dataset_train), len(clean_dataset_train[0]), len(poisoned_test))

    # 显示图片
    plt.figure()
    plt.imshow(poisoned_dataset_train.data[1][0])
    #保存图片为eps格式
    plt.savefig('./save/atk_cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()

    return poisoned_dataset_train, clean_dataset_train, poisoned_test

class LocalUpdate_backdoor(object):
    def __init__(self, args, dataset=None,test_dataset = None, idxs=None, Forgetting_degree = 0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.Forget_degree = Forgetting_degree
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        self.contribution_degree = 1 - Forgetting_degree
        self.Remain_ep = int(args.local_ep - args.local_ep * Forgetting_degree)
        self.Forget_ep = int(args.local_ep * Forgetting_degree)

    # 遗忘网络训练
    def Forget_train(self, Remainnet, Forgetnet, optimizer_Forget, Forgetnet_ep, prev_loss):
        Forgetnet.train()
        # 如果遗忘轮数为0，返回无限大
        if Forgetnet_ep == 0:
            return float('inf'), float('inf'), float('inf')
        # 记录初始网络参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        last_epoch_loss = []
        last_epoch_loss_avg = 0

        for epoch in range(Forgetnet_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Forget.step()
                if self.args.verbose and batch_idx % 100 == 0:
                    print('Forget Update Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, Forgetnet_ep, batch_idx * len(images), len(self.ldr_train.dataset),
                                             100. * batch_idx / len(self.ldr_train), loss.item()))
                if epoch == Forgetnet_ep - 1:
                    last_epoch_loss.append(loss.item())

        # 计算最后一轮平均损失值
        last_epoch_loss_avg = sum(last_epoch_loss) / len(last_epoch_loss)

        # 记录训练后网络参数
        Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

        # 计算模型差异度
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim = cos(w_glob, w_train)
        cos_sim = (cos_sim + 1.0) / 2.0
        diff = 1 - cos_sim

        # 计算差异阈值
        diff_threshold = self.contribution_degree / self.Forget_degree * diff
        # 计算损失阈值
        loss_threshold = self.contribution_degree / self.Forget_degree * (prev_loss - last_epoch_loss_avg)
        # 返回模型差异度阈值和损失值差异阈值
        return diff_threshold, loss_threshold, last_epoch_loss_avg

    def record(self, Remainnet, Forgetnet):
        record_loss = []
        Remainnet.train()
        Forgetnet.train()
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            temp = Remainnet(images)
            log_probs = Forgetnet(temp)
            record_loss.append(self.loss_func(log_probs, labels).item())
        return sum(record_loss) / len(record_loss)

    # 保留网络训练
    def Remain_train(self, Remainnet, Forgetnet, optimizer_Remain, Remainnet_ep, diff_threshold, loss_threshold,
                     prev_loss):
        # 记录初始网路参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        # 训练
        Remainnet.train()
        Remainnet_avg_loss = 0
        for epoch in range(Remainnet_ep):
            batch_loss = []
            # 存储网络参数

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                batch_loss.append(loss.item())

                # 记录平均损失值
                Remainnet_avg_loss = sum(batch_loss) / len(batch_loss)
                loss_diff = prev_loss - Remainnet_avg_loss
                # 每100次判断是否达到损失值阈值

                # 记录训练完成后网络参数
                Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
                Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
                w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

                # 差异度
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sim = cos(w_glob, w_train)
                cos_sim = (cos_sim + 1.0) / 2.0
                diff = 1 - cos_sim

                # 判断是否达到差异度阈值，跳出当前循环
                if diff > diff_threshold:
                    print("diff:{:.4f}> diff_thresholdf:{:.4f}".format(diff, diff_threshold))
                    return Remainnet_avg_loss
                if batch_idx > 10 and loss_diff > loss_threshold:
                    print("loss_difff:{:.4f} > loss_thresholdf:{:.4f}".format(loss_diff, loss_threshold))
                    return Remainnet_avg_loss
                if self.args.verbose and batch_idx % 100 == 0:
                    print(
                        'Remainnet Update Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f} diff/diff_threshold:{:.4f}/{:.4f}\tloss_diff/loss_threshold:{:.4f}/{:.4f}'.format(
                            epoch, Remainnet_ep, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item(), diff,
                            diff_threshold, loss_diff, loss_threshold))

        return Remainnet_avg_loss

    def Normaltrain(self, Remainnet, Forgetnet, optimizer_Remain, optimizer_Forget):
        Remainnet.train()
        Forgetnet.train()
        loss_ = []
        loss_avg = 0
        for epoch in range(self.args.cycles):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_Remain.zero_grad()
                optimizer_Forget.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                optimizer_Forget.step()
                loss_.append(loss.item())
                if self.args.verbose and batch_idx % 100 == 0:
                    print('Normal Update Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, self.args.cycles, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item()))
            loss_avg = sum(loss_) / len(loss_)
        return loss_avg
        # 客户端训练

    def Localtrain(self, Remainnet, Forgetnet):
        # 初始化优化器
        optimizer_Remain = torch.optim.SGD(Remainnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_Forget = torch.optim.SGD(Forgetnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 先记录初始损失值
        prev_loss = self.record(Remainnet, Forgetnet)
        print('prev_loss:{:.4f}'.format(prev_loss))
        loss = []
        acc_poison = []
        acc_clean = []
        # if self.Forget_degree == 0:
        #     loss_avg = self.Normaltrain(Remainnet,Forgetnet,optimizer_Remain,optimizer_Forget)
        # else:
        Remainnet.train()
        Forgetnet.train()
        for cycle in range(self.args.cycles):
            print("Cycle: {}".format(cycle + 1))
            # 训练遗忘网络
            prev_loss = self.record(Remainnet, Forgetnet)
            if self.Forget_degree != 0:
                print("Training Forget network")
            diff_threshold, loss_threshold, loss_train = self.Forget_train(Remainnet,
                                                                           Forgetnet, optimizer_Remain, self.Forget_ep,
                                                                           prev_loss)
            if loss_train != float('inf'):
                loss.append(loss_train)
            # 差异度阈值， 损失值阈值 ，当前平均损失值
            if self.Forget_degree != 0:
                print("diff_threshold: {:.4f}, loss_threshold: {:.4f}, Loss: {:.4f}".format(diff_threshold,
                                                                                            loss_threshold, loss_train))

                # 再训练子网络
                print("Training Remainnet")
            prev_loss = self.Remain_train(Remainnet, Forgetnet, optimizer_Forget, self.Remain_ep,
                                          diff_threshold, loss_threshold, loss_train)
            loss.append(prev_loss)
            # poison_acc, poison_loss = test_img(Remainnet, Forgetnet,self.test_dataset, self.args)
            # clean_acc, clean_loss = test_img(Remainnet, Forgetnet, self.dataset,self.args)
            # acc_poison.append(poison_acc)
        # 计算平均损失值
        loss_avg = sum(loss) / len(loss)
        # 返回网络和损失值
        return Remainnet.state_dict(), Forgetnet.state_dict(),loss_avg
class LocalUpdate_A_atk(object):
    def __init__(self, args, dataset=None, idxs=None, Forgetting_degree=0,batch_para = 1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.Forget_degree = Forgetting_degree
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        self.contribution_degree = 1 - Forgetting_degree
        # self.Remain_ep = int(args.local_ep * self.contribution_degree)
        # self.Forget_ep = int(args.local_ep * Forgetting_degree)
        self.batch_num = len(self.ldr_train.dataset) / self.args.local_bs
        self.batch_num_forget = int(self.batch_num * batch_para * self.Forget_degree )
        self.batch_num_remain = int(self.batch_num * self.contribution_degree * batch_para)
    # 遗忘网络训练
    def Forget_train(self, Remainnet, Forgetnet, optimizer_Forget, prev_loss):
        Forgetnet.train()
        Remainnet.train()
        # 如果遗忘轮数为0，返回无限大
        if self.batch_num_forget == 0:
            return float('inf'), float('inf'), float('inf')
        # 记录初始网络参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        last_epoch_loss = []
        last_epoch_loss_avg = 0
        batch_sum = 0
        Forgetnet_ep = 0
        while batch_sum < self.batch_num_forget:
            Forgetnet_ep += 1
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_sum > self.batch_num_forget:
                    break
                batch_sum += 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Forget.step()
                if self.args.verbose and batch_idx % 5 == 0:
                    print('Forget Update batch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_sum,self.batch_num_forget , batch_idx * len(images), len(self.ldr_train.dataset),
                                             100. * batch_idx / len(self.ldr_train), loss.item()))
                last_epoch_loss.append(loss.item())
        # 计算最后一轮平均损失值
        last_epoch_loss_avg = sum(last_epoch_loss) / len(last_epoch_loss)

        # 记录训练后网络参数
        Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

        # 计算模型差异度
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim = cos(w_glob, w_train)
        cos_sim = (cos_sim + 1.0) / 2.0
        diff = 1 - cos_sim

        # 计算差异阈值
        diff_threshold = self.contribution_degree / self.Forget_degree * diff
        # 计算损失阈值
        loss_threshold = self.contribution_degree / self.Forget_degree * (last_epoch_loss[0] - last_epoch_loss[-1])
        # 返回模型差异度阈值和损失值差异阈值
        return diff_threshold, loss_threshold, last_epoch_loss_avg

    def record(self, Remainnet, Forgetnet):
        record_loss = []
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            temp = Remainnet(images)
            log_probs = Forgetnet(temp)
            record_loss.append(self.loss_func(log_probs, labels).item())
        return sum(record_loss) / len(record_loss)

    # 保留网络训练
    def Remain_train(self, Remainnet, Forgetnet, optimizer_Remain, diff_threshold, loss_threshold,
                     prev_loss):
        # 记录初始网路参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        # 训练
        Remainnet.train()
        Forgetnet.train()
        Remainnet_avg_loss = 0
        batch_sum = 0
        epoch = 0
        while batch_sum < self.batch_num_remain:
            epoch += 1
            batch_loss = []
            # 存储网络参数
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_sum >= self.batch_num_remain:
                    return Remainnet_avg_loss
                batch_sum += 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                batch_loss.append(loss.item())

                # 记录平均损失值
                Remainnet_avg_loss = sum(batch_loss) / len(batch_loss)
                # loss_diff = prev_loss - Remainnet_avg_loss
                loss_diff = batch_loss[0] - batch_loss[-1]
                # 每100次判断是否达到损失值阈值
                if batch_idx != 0 and batch_idx % 5 == 0 and loss_diff > loss_threshold:
                    print("loss_difff:{:.4f} > loss_thresholdf:{:.4f}".format(loss_diff, loss_threshold))
                    return Remainnet_avg_loss

                # 记录训练完成后网络参数
                Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
                Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
                w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

                # 差异度
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sim = cos(w_glob, w_train)
                cos_sim = (cos_sim + 1.0) / 2.0
                diff = 1 - cos_sim

                # 判断是否达到差异度阈值，跳出当前循环
                if diff > diff_threshold:
                    print("diff:{:.4f}> diff_thresholdf:{:.4f}".format(diff, diff_threshold))
                    return Remainnet_avg_loss

                if self.args.verbose and batch_idx % 5 == 0:
                    print(
                        'Remainnet Update batch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f} diff/diff_threshold:{:.4f}/{:.4f}\tloss_diff/loss_threshold:{:.4f}/{:.4f}'.format(
                            batch_sum,self.batch_num_remain, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item(), diff,
                            diff_threshold, loss_diff, loss_threshold))
        return Remainnet_avg_loss

    def Normaltrain(self, Remainnet, Forgetnet, optimizer_Remain, optimizer_Forget):
        Remainnet.train()
        Forgetnet.train()
        loss_ = []
        loss_avg = 0
        for epoch in range(self.args.cycles):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_Remain.zero_grad()
                optimizer_Forget.zero_grad()
                temp = Remainnet(images)
                log_probs = Forgetnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                optimizer_Forget.step()
                loss_.append(loss.item())
                if self.args.verbose and batch_idx % 100 == 0:
                    print('Normal Update Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, self.args.cycles, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item()))
            loss_avg = sum(loss_) / len(loss_)
        return loss_avg
        # 客户端训练

    def Localtrain(self, Remainnet, Forgetnet):
        # 初始化优化器
        optimizer_Remain = torch.optim.SGD(Remainnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_Forget = torch.optim.SGD(Forgetnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 先记录初始损失值
        prev_loss = self.record(Remainnet, Forgetnet)
        print('prev_loss:{:.4f}'.format(prev_loss))
        loss = []
        # if self.Forget_degree == 0:
        #     loss_avg = self.Normaltrain(Remainnet,Forgetnet,optimizer_Remain,optimizer_Forget)
        # else:
        for cycle in range(self.args.cycles):
            print("Cycle: {}".format(cycle + 1))
            # 训练遗忘网络
            if self.Forget_degree != 0:
                print("Training Forget network")
            diff_threshold, loss_threshold, loss_train = self.Forget_train(Remainnet,
                                                                           Forgetnet, optimizer_Remain,
                                                                           prev_loss)
            if loss_train != float('inf'):
                loss.append(loss_train)
            # 差异度阈值， 损失值阈值 ，当前平均损失值
            if self.Forget_degree != 0:
                print("diff_threshold: {:.4f}, loss_threshold: {:.4f}, Loss: {:.4f}".format(diff_threshold,
                                                                                            loss_threshold, loss_train))

                # 再训练子网络
                print("Training Remainnet")
            prev_loss = self.Remain_train(Remainnet, Forgetnet, optimizer_Forget,
                                          diff_threshold, loss_threshold, loss_train)
            loss.append(prev_loss)
        # 计算平均损失值
        loss_avg = sum(loss) / len(loss)
        # 返回网络和损失值
        return Remainnet.state_dict(), Forgetnet.state_dict(), loss_avg

class LocalUpdate_A_atk_Swap(object):
    def __init__(self, args, dataset=None, idxs=None, Forgetting_degree=0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.Forget_degree = Forgetting_degree
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        self.contribution_degree = 1 - Forgetting_degree
        # self.Remain_ep = int(args.local_ep * self.contribution_degree)
        # self.Forget_ep = int(args.local_ep * Forgetting_degree)
        self.batch_num = len(self.ldr_train.dataset) / self.args.local_bs
        self.batch_num_forget = int(self.batch_num * 1)
        self.batch_num_remain = int(self.batch_num * self.contribution_degree / self.Forget_degree * 1)
    # 遗忘网络训练
    def Forget_train(self, Remainnet, Forgetnet, optimizer_Forget, prev_loss):
        Forgetnet.train()
        Remainnet.train()
        # 如果遗忘轮数为0，返回无限大
        if self.batch_num_forget == 0:
            return float('inf'), float('inf'), float('inf')
        # 记录初始网络参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        last_epoch_loss = []
        last_epoch_loss_avg = 0
        batch_sum = 0
        Forgetnet_ep = 0
        while batch_sum < self.batch_num_forget:
            Forgetnet_ep += 1
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_sum > self.batch_num_forget:
                    break
                batch_sum += 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                # temp = Remainnet(images)
                # log_probs = Forgetnet(temp)
                temp = Forgetnet(images)
                log_probs = Remainnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Forget.step()
                if self.args.verbose and batch_idx % 5 == 0:
                    print('Forget Update batch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_sum,self.batch_num_forget , batch_idx * len(images), len(self.ldr_train.dataset),
                                             100. * batch_idx / len(self.ldr_train), loss.item()))
                last_epoch_loss.append(loss.item())
        # 计算最后一轮平均损失值
        last_epoch_loss_avg = sum(last_epoch_loss) / len(last_epoch_loss)

        # 记录训练后网络参数
        Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

        # 计算模型差异度
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_sim = cos(w_glob, w_train)
        cos_sim = (cos_sim + 1.0) / 2.0
        diff = 1 - cos_sim

        # 计算差异阈值
        diff_threshold = self.contribution_degree / self.Forget_degree * diff
        # 计算损失阈值
        loss_threshold = self.contribution_degree / self.Forget_degree * (last_epoch_loss[0] - last_epoch_loss[-1])
        # 返回模型差异度阈值和损失值差异阈值
        return diff_threshold, loss_threshold, last_epoch_loss_avg

    def record(self, Remainnet, Forgetnet):
        record_loss = []
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            temp = Forgetnet(images)
            log_probs = Remainnet(temp)
            record_loss.append(self.loss_func(log_probs, labels).item())
        return sum(record_loss) / len(record_loss)

    # 保留网络训练
    def Remain_train(self, Remainnet, Forgetnet, optimizer_Remain, diff_threshold, loss_threshold,
                     prev_loss):
        # 记录初始网路参数
        Remainnet_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
        Forgetnet_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
        w_glob = torch.cat([Remainnet_vec, Forgetnet_vec])

        # 训练
        Remainnet.train()
        Forgetnet.train()
        Remainnet_avg_loss = 0
        batch_sum = 0
        epoch = 0
        while batch_sum < self.batch_num_remain:
            epoch += 1
            batch_loss = []
            # 存储网络参数
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_sum >= self.batch_num_remain:
                    return Remainnet_avg_loss
                batch_sum += 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                Remainnet.zero_grad()
                Forgetnet.zero_grad()
                temp = Forgetnet(images)
                log_probs = Remainnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                batch_loss.append(loss.item())

                # 记录平均损失值
                Remainnet_avg_loss = sum(batch_loss) / len(batch_loss)
                # loss_diff = prev_loss - Remainnet_avg_loss
                loss_diff = batch_loss[0] - batch_loss[-1]
                # 每100次判断是否达到损失值阈值
                if batch_idx != 0 and batch_idx % 5 == 0 and loss_diff > loss_threshold:
                    print("loss_difff:{:.4f} > loss_thresholdf:{:.4f}".format(loss_diff, loss_threshold))
                    return Remainnet_avg_loss

                # 记录训练完成后网络参数
                Forgetnet_train_vec = torch.cat([param.view(-1) for param in Forgetnet.parameters()])
                Remainnet_train_vec = torch.cat([param.view(-1) for param in Remainnet.parameters()])
                w_train = torch.cat([Remainnet_train_vec, Forgetnet_train_vec])

                # 差异度
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sim = cos(w_glob, w_train)
                cos_sim = (cos_sim + 1.0) / 2.0
                diff = 1 - cos_sim

                # 判断是否达到差异度阈值，跳出当前循环
                if diff > diff_threshold:
                    print("diff:{:.4f}> diff_thresholdf:{:.4f}".format(diff, diff_threshold))
                    return Remainnet_avg_loss

                if self.args.verbose and batch_idx % 5 == 0:
                    print(
                        'Remainnet Update batch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f} diff/diff_threshold:{:.4f}/{:.4f}\tloss_diff/loss_threshold:{:.4f}/{:.4f}'.format(
                            batch_sum,self.batch_num_remain, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item(), diff,
                            diff_threshold, loss_diff, loss_threshold))
        return Remainnet_avg_loss

    def Normaltrain(self, Remainnet, Forgetnet, optimizer_Remain, optimizer_Forget):
        Remainnet.train()
        Forgetnet.train()
        loss_ = []
        loss_avg = 0
        for epoch in range(self.args.cycles):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_Remain.zero_grad()
                optimizer_Forget.zero_grad()
                temp = Forgetnet(images)
                log_probs = Remainnet(temp)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_Remain.step()
                optimizer_Forget.step()
                loss_.append(loss.item())
                if self.args.verbose and batch_idx % 100 == 0:
                    print('Normal Update Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, self.args.cycles, batch_idx * len(images), len(self.ldr_train.dataset),
                                                 100. * batch_idx / len(self.ldr_train), loss.item()))
            loss_avg = sum(loss_) / len(loss_)
        return loss_avg
        # 客户端训练

    def Localtrain(self, Remainnet, Forgetnet):
        # 初始化优化器
        optimizer_Remain = torch.optim.SGD(Remainnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_Forget = torch.optim.SGD(Forgetnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 先记录初始损失值
        prev_loss = self.record(Remainnet, Forgetnet)
        print('prev_loss:{:.4f}'.format(prev_loss))
        loss = []
        # if self.Forget_degree == 0:
        #     loss_avg = self.Normaltrain(Remainnet,Forgetnet,optimizer_Remain,optimizer_Forget)
        # else:
        for cycle in range(self.args.cycles):
            print("Cycle: {}".format(cycle + 1))
            # 训练遗忘网络
            if self.Forget_degree != 0:
                print("Training Forget network")
            diff_threshold, loss_threshold, loss_train = self.Forget_train(Remainnet,
                                                                           Forgetnet, optimizer_Remain,
                                                                           prev_loss)
            if loss_train != float('inf'):
                loss.append(loss_train)
            # 差异度阈值， 损失值阈值 ，当前平均损失值
            if self.Forget_degree != 0:
                print("diff_threshold: {:.4f}, loss_threshold: {:.4f}, Loss: {:.4f}".format(diff_threshold,
                                                                                            loss_threshold, loss_train))

                # 再训练子网络
                print("Training Remainnet")
            prev_loss = self.Remain_train(Remainnet, Forgetnet, optimizer_Forget,
                                          diff_threshold, loss_threshold, loss_train)
            loss.append(prev_loss)
        # 计算平均损失值
        loss_avg = sum(loss) / len(loss)
        # 返回网络和损失值
        return Remainnet.state_dict(), Forgetnet.state_dict(), loss_avg