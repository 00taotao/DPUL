import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
# from multiprocessing import Pool, Manager
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torchvision import datasets
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
from transformers import ViTImageProcessor, ViTForImageClassification

from models.Autoencoder import AutoEncoder, VAE, VAE1
from models.Fed import FedAvg
from models.Nets import CNNMnist, CNNCifar, MLP
from models.Update import LocalUpdate
from models.VAE import BetaVAE_H
from models.Vectomodel import vectomodel_vit
from models.test import test_img, test_vit
from utils.load_datasets import TinyImageNet, CINIC10
from utils.options import args_parser
from utils.sample import cifar100_noniid, cinic10_noniid, imagenet_tiny_noniid
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.seed import set_seed
from tqdm import tqdm
import torch.multiprocessing as mp
import time
class Trainer:
    def __init__(self, args, autoencoders, M_slices, MU_slices, optimizers):
        self.args = args
        self.epochs = args.AE_epochs
        self.autoencoders = autoencoders
        self.M_slices = M_slices
        self.MU_slices = MU_slices
        self.optimizers = optimizers
        self.beta = args.beta
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train_slice(self, j):
        torch.cuda.set_device(self.args.gpu)
        for epoch in tqdm(range(self.epochs)):
            for i in range(len(self.M_slices) - 1):
                output, mu, logvar = self.autoencoders[j](self.M_slices[i][j])
                loss = nn.MSELoss()(output, self.MU_slices[i][j])
                loss = loss - self.beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoders[j].parameters(), max_norm=1.0)
                step = self.optimizers[j].step()
                self.optimizers[j].zero_grad()
        # 测量相似度
        for i in range(len(self.M_slices)):
            output, _, _ = self.autoencoders[j](self.M_slices[i][j])
            print(f'slice{j},',self.cos(output, self.MU_slices[i][j]))
        return
def main():
    # 初始化参数
    mp.set_start_method('spawn')
    args = args_parser()
    args.gpu = 0
    args.epochs = 50
    args.num_users = 10
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.AE_epochs = 2000
    args.slices = 10
    args.lr = 5e-5
    args.iid = False
    args.beta = 0.5
    args.dataset = 'cifar10'
    args.model = 'Vits'
    FLepoch = 50
    if not os.path.exists('./save/FUL/{}/N{}/E{}'.format(args.dataset, args.num_users, FLepoch)):
        os.makedirs('./save/FUL/{}/N{}/E{}'.format(args.dataset, args.num_users, FLepoch))
    # 加载数据集
    if args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)
    elif args.dataset == 'imagenet-tiny':
        trans_imagetiny = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                              transforms.Normalize( (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])
        dataset_train = TinyImageNet('./data/tiny-imagenet-200', train=True, transform=trans_imagetiny)
        dataset_test = TinyImageNet('./data/tiny-imagenet-200', train=False, transform=trans_imagetiny)
        dict_users = imagenet_tiny_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cinic10':
        trans_cinic = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(),
             transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])
        dataset_train = CINIC10('./data/cinic10', split='train', transform=trans_cinic)
        dataset_test = CINIC10('./data/cinic10', split='test', transform=trans_cinic)
        dict_users = cinic10_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # 加载模型
    if args.model == 'Vits' and args.dataset == 'cifar10':
        model_path = "./data/Vit-small-patch16-224"
        feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 10)  # CIFAR-10有10个类别
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["classifier", ])
        net_glob = get_peft_model(model, lora_config).to(args.device)
    elif args.model == 'Vitb' and args.dataset == 'cifar100':
        model_path = "./data/Vit-base-patch16-224-in21k"
        feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 100)
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["classifier", ])
        net_glob = get_peft_model(model, lora_config).to(args.device)
    elif args.model == 'Vitl' and args.dataset == 'imagenet-tiny':
        model_path = './data/Vit-large-patch16-224-in21k'
        feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 200)
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["classifier", ])
        net_glob = get_peft_model(model, lora_config).to(args.device)
    elif args.model == 'Deitb' and args.dataset == 'cinic10':
        model_path = './data/deit-base-patch16-224'
        feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 10)
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["classifier", ])
        net_glob = get_peft_model(model, lora_config).to(args.device)


    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    net_test = copy.deepcopy(net_glob)
    # 复制全局模型
    w_glob = net_glob.state_dict()
    M = []
    MU = []
    U = []
    # 加载数据
    for i in range(FLepoch):
        M.append(torch.load('./save/FL/{}/{}/N{}/E{}/M{}.pth'.format(args.dataset,args.lr, args.num_users,args.epochs,i),
                            map_location=args.device))
        U.append(torch.load('./save/FL/{}/{}/N{}/E{}/U{}.pth'.format(args.dataset, args.lr, args.num_users, args.epochs, i),
                            map_location=args.device))

    client_data_sizes = [len(dict_users[i]) for i in range(args.num_users)]
    sum_client_data_sizes = sum(client_data_sizes)
    delta_M = [M[i] - M[i - 1] for i in range(1, len(M))]
    delta_U = [U[i] - M[i - 1] for i in range(1, len(U))]
    MU = [(M[i] * sum_client_data_sizes  - client_data_sizes[0] * U[i])/(sum_client_data_sizes - client_data_sizes[0]) for i in range(len(M))]
    # 复制MU
    MU_old = [MU[i].clone() for i in range(len(MU))]

    cnt = 0
    for i,(m,u) in enumerate(zip(delta_M,delta_U)):
        for j,(mm,uu) in enumerate(zip(m,u)):
            if abs(mm)<abs(client_data_sizes[0]/sum_client_data_sizes * uu):
                for k in range(i,len(MU)):
                    if k == 0:
                        continue
                    MU[k][j] = MU[k-1][j]

    # 测试MU
    net_glob = vectomodel_vit(MU_old[-1], net_glob)
    acc, loss = test_vit(net_glob, dataset_test, args)
    # 保存acc, loss为txt文件
    with open('./save/FUL/{}/N{}/E{}/MU_old.txt'.format(args.dataset, args.num_users, FLepoch), 'w') as f:
        f.write(str(acc) + '\n')
        f.write(str(loss) + '\n')
    net_glob = vectomodel_vit(MU[-1], net_glob)
    acc, loss = test_vit(net_glob, dataset_test, args)
    # 保存acc, loss为txt文件
    with open('./save/FUL/{}/N{}/E{}/MU_process.txt'.format(args.dataset, args.num_users, FLepoch), 'w') as f:
        f.write(str(acc) + '\n')
        f.write(str(loss) + '\n')

    # 用遗忘客户端参数和全局参数训练自编码器
    n = len(M)
    # 将 M 和 MU 切成片，并调用detach方法
    M_slices = [[M[i][j::args.slices].detach() for j in range(args.slices)] for i in range(n)]
    MU_slices = [[MU[i][j::args.slices].detach() for j in range(args.slices)] for i in range(n)]
    autoencoders = [BetaVAE_H(z_dim=1, seq_length=len(M_slices[0][i])).to(args.device) for i in range(args.slices)]
    # optimizers = [torch.optim.Adam(autoencoder.parameters(), lr=0.001) for autoencoder in autoencoders]
#   SGD 优化器
    optimizers = [torch.optim.SGD(autoencoder.parameters(), lr=0.1, momentum=0.9) for autoencoder in autoencoders]
    trainer = Trainer(args, autoencoders, M_slices, MU_slices, optimizers)
    # 统计时间
    MU1 = MU[-1]
    M0 = M[0]
    M1=M[-1]
    # 清空MU和M
    MU = []
    M = []
    start = time.time()
    with mp.Pool(processes=args.slices) as pool:
        pool.map(trainer.train_slice, [j for j in range(args.slices)])
    end = time.time()
    print('time:', end - start)
    # test VAE 测试自编码器，只测最后一轮
    tmp = copy.deepcopy(M0)
    cos_list = []
    for i in range(args.slices):
        output, _, _ = autoencoders[i](M_slices[-1][i])
        tmp_clone = tmp.clone()
        tmp_clone[i::args.slices] = output
        tmp = tmp_clone
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_list.append(cos(output, MU_slices[-1][i]))
        print(f'slice{i},', cos(output, MU_slices[-1][i]))
    print(cos(tmp, MU1))
    M_last = M_slices[-1]
    M_test = copy.deepcopy(M1)
    for epoch in range(1):
        for j in range(args.slices):
            output, mu, logvar = autoencoders[j](M_last[j])  # loss等于所有的欧氏距离
            M_last[j] = output.detach().clone()
            # 测试cos
            print(cos(M_last[j], MU_slices[-1][j]))
    # 将M_last 拼回 M_test
    M_test_new = M_test.clone().detach()
    for j in range(args.slices):
        M_test_new[j::args.slices] = M_last[j].detach().clone()
    # 计算相似度
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    print(cos(M_test_new, MU1))

    net_glob = vectomodel_vit(M_test_new,net_glob)
    acc,loss = test_vit(net_glob, dataset_test, args)
    # 保存acc, loss为txt文件
    with open('./save/FUL/{}/N{}/E{}/MU_vae.txt'.format(args.dataset, args.num_users, FLepoch), 'w') as f:
        f.write(str(acc) + '\n')
        f.write(str(loss) + '\n')

    # save VAE 保存10个自编码器
    for i in range(args.slices):
        torch.save(autoencoders[i].state_dict(),
                   './save/FUL/{}/N{}/E{}/VAE{}_{}.pth'.format(args.dataset, args.num_users, FLepoch, args.AE_epochs,                                                           i))
    # save cos
    with open('./save/FUL/{}/N{}/E{}/cos{}.txt'.format(args.dataset, args.num_users, FLepoch, args.AE_epochs), 'w') as f:
        for cos in cos_list:
            f.write(str(cos.item()) + '\n')

if __name__ == '__main__':
    main()

    # 统计时间
