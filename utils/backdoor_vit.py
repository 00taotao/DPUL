import io
import pickle

import numpy as np
import art
import pandas as pd
import torchvision
from PIL import Image
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import load_mnist, preprocess, to_categorical
## load_cifar10
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

import os

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




def load_cifar10(data_dir):
    images = []
    labels = []

    for batch in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{batch}')

        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            images.append(data[b'data'])
            labels.extend(data[b'labels'])  # CIFAR-10 使用 'labels' 而非 'fine_labels'

    images = np.concatenate(images)
    images = images.reshape(-1,3, 32, 32)
    images = images.astype('float32') / 255.0  # 归一化到 [0, 1] 范围内
    labels = np.array(labels)

    return images, labels
def generate_Cifar10_backdoor(num_parties=2, scale=1):
    # 加载数据
    # (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)
    path = "../data/cifar/cifar-10-batches-py"
    x_raw, y_raw= load_cifar10(path)
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    # x_test, y_test = preprocess(x_raw_test, y_raw_test)



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

    backdoor = PoisoningAttackBackdoor((lambda x: insert_image(x, mode='RGB')))
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
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(224),
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

    backdoor = PoisoningAttackBackdoor((lambda x: insert_image(x, mode='RGB',backdoor_path='../data/backdoor/6x6.png')))
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
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(224),
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


def load_cinic10(data_dir):
    images = []
    labels = []

    # 加载训练数据
    train_path = f'{data_dir}/train-00000-of-00001.parquet'
    train_data = pd.read_parquet(train_path)

    # 提取图像和标签
    for index, row in train_data.iterrows():
        img_dict = row['image']  # 假设图像存储在 'image' 列中
        img_bytes = img_dict['bytes']  # 从字典中提取字节数据
        image = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))  # 转为 NumPy 格式 (H, W, 3)
        images.append(image)
        labels.append(row['label'])  # 提取标签

    # 转换为 NumPy 数组并调整维度
    images = np.array(images).astype('float32') / 255.0  # 归一化到 [0, 1]
    images = images.transpose(0, 3, 1, 2)  # 调整维度为 (N, 3, 32, 32)
    labels = np.array(labels)
    return images, labels

def generate_Cinic10_backdoor(num_parties=2, scale=1):
    # 加载数据
    # (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)
    path = "../data/cinic10/data"
    x_raw, y_raw= load_cinic10(path)
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw)
    # x_test, y_test = preprocess(x_raw_test, y_raw_test)



    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    x_train = x_train.transpose(0, 2, 3, 1)

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(90000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((90000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本

    backdoor = PoisoningAttackBackdoor((lambda x: insert_image(x, mode='RGB')))
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
    plt.imshow(poison_before.data[2][0])
    #保存图片为eps格式
    plt.savefig('./save/cinic10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
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
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),  # R,G,B每层的归一化用到的均值和方差
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
    plt.imshow(poisoned_dataset_train.data[2][0])
    #保存图片为eps格式
    plt.savefig('./save/atk_cifar10.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()

    return poisoned_dataset_train, clean_dataset_train, poisoned_test


def load_imagenettiny(data_dir):
    """
    加载 Tiny ImageNet 数据集
    :param data_dir: 数据集根目录路径
    :param train: 是否加载训练集 (True: 加载训练集, False: 加载验证集)
    :return: 图像数据和标签
    """
    images = []
    labels = []
    dataset_dir = os.path.join(data_dir, "train")
    class_names = sorted(os.listdir(dataset_dir))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    # 加载训练数据
    for class_name in class_names:
        class_folder = os.path.join(dataset_dir, class_name, "images")
        if not os.path.isdir(class_folder):
            continue
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            if not file_name.endswith(".JPEG"):
                continue  # 过滤非图像文件
            try:
                image = np.array(Image.open(file_path).convert('RGB'))  # 转为 NumPy 格式 (H, W, 3)
                images.append(image)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"无法加载图像 {file_path}: {e}")
    images = np.array(images).astype('float32') / 255.0  # 归一化到 [0, 1]
    labels = np.array(labels)
    # 如果需要，可以调整维度，例如用于 PyTorch 的 (N, 3, H, W)
    images = images.transpose(0, 3, 1, 2)  # 调整维度为 (N, 3, H, W)
    return images, labels

def generate_ImageNetTiny_backdoor(num_parties=2, scale=1):
    # 加载数据
    # (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)
    path = "../data/tiny-imagenet-200"
    x_raw, y_raw= load_imagenettiny(path)
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw, nb_classes=200)
    # x_test, y_test = preprocess(x_raw_test, y_raw_test)



    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    x_train = x_train.transpose(0, 2, 3, 1)

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(90000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((90000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本

    backdoor = PoisoningAttackBackdoor((lambda x: insert_image(x, mode='RGB',backdoor_path = '../data/backdoor/20x20.png')))
    example_target = np.array([0] * 199 + [1])   # 后门攻击标签
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
    plt.imshow(poison_before.data[2][0])
    #保存图片为eps格式
    plt.savefig('./save/imagenettiny.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
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
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),  # R,G,B每层的归一化用到的均值和方差
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
    plt.imshow(poisoned_dataset_train.data[2][0])
    #保存图片为eps格式
    plt.savefig('./save/imagenettiny.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()

    return poisoned_dataset_train, clean_dataset_train, poisoned_test

def generate_ImageNetTiny_backdoor(num_parties=2, scale=1):
    # 加载数据
    # (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)
    path = "../data/tiny-imagenet-200"
    x_raw, y_raw= load_imagenettiny(path)
    #预处理
    x_train, y_train = preprocess(x_raw, y_raw, nb_classes=200)
    # x_test, y_test = preprocess(x_raw_test, y_raw_test)



    # 打乱
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    x_train = x_train.transpose(0, 2, 3, 1)

    num_parties = num_parties  # 5个客户端
    scale = scale  # 数量
    num_samples_erased_party = int(90000 / num_parties * scale)  # 后门攻击客户端的样本数量
    num_samples_per_party = int((90000 - num_samples_erased_party) / (num_parties - 1))  # 剩余每个客户端的数据数量
    x_train_party = x_train[0:num_samples_erased_party]  # 后门攻击客户端的训练集
    y_train_party = y_train[0:num_samples_erased_party]  # 后门攻击客户端的标签

    # 选择后门攻击样本

    backdoor = PoisoningAttackBackdoor((lambda x: insert_image(x, mode='RGB',backdoor_path = '../data/backdoor/20x20.png')))
    example_target = np.array([0] * 199 + [1])   # 后门攻击标签
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
    plt.imshow(poison_before.data[2][0])
    #保存图片为eps格式
    plt.savefig('./save/imagenettiny.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
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
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),  # R,G,B每层的归一化用到的均值和方差
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
    plt.imshow(poisoned_dataset_train.data[2][0])
    #保存图片为eps格式
    plt.savefig('./save/imagenettiny.eps', format='eps',bbox_inches='tight',pad_inches=0.0)
    plt.show()

    return poisoned_dataset_train, clean_dataset_train, poisoned_test