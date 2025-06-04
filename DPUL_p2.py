import copy
import os
import time

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

from models.Autoencoder import AutoEncoder, VAE, AutoEncoder2, AutoEncoder1, ConvAutoEncoder, CAE_hidden
from models.Fed import FedAvg
from models.Nets import CNNMnist, CNNCifar, MLP, LeNet5
from models.Update import LocalUpdate, LocalUpdate_vit
from models.VAE import BetaVAE_H
from models.Vectomodel import vectomodel_vit
from models.test import test_img, test_vit
from utils.load_datasets import TinyImageNet, CINIC10
from utils.options import args_parser
from utils.sample import cifar100_noniid, imagenet_tiny_noniid, cinic10_noniid
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.seed import set_seed

if __name__ == '__main__':
    # 初始化参数
    set_seed(0)
    args = args_parser()
    args.epochs = 50
    args.num_users = 10
    args.dataset = 'cifar10'
    args.model = 'Vits'
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.AE_epochs = 2000
    slices = 10
    args.lr = 5e-5
    learning_rate = 5e-5
    args.post_epochs = 50
    # 加载数据集
    if not os.path.exists('./save/{}/N{}/E{}/compare'.format(args.dataset, args.num_users, args.epochs)):
        os.makedirs('./save/{}/N{}/E{}/compare'.format(args.dataset, args.num_users, args.epochs))
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
                                              transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])
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
        model_path = './data/Vit-large-patch16-224'
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
    # 加载数据
    for i in range(args.epochs):
        M.append(torch.load('./save/FL/{}/{}/N{}/E{}/M{}.pth'.format(args.dataset, args.lr,args.num_users,args.epochs, i),map_location=args.device))
        MU.append(torch.load('./save/FL/{}/{}/N{}/E{}/MU{}.pth'.format(args.dataset, args.lr,args.num_users,args.epochs, i),map_location=args.device))
    # 载入准确率
    with open('./save/FL/{}/{}/N{}/E{}/val_acc.txt'.format(args.dataset, args.lr,args.num_users,args.epochs), 'r') as f:
        fl_acc = [float(i) for i in f.readlines()]

    # 用遗忘客户端参数和全局参数训练自编码器
    net_test = vectomodel_vit(MU[0],net_test)
    n = len(M)
    # 将 M 和 MU 切成 10 片
    M_slices = [[M[i][j::slices] for j in range(slices)] for i in range(n)]
    MU_slices = [[MU[i][j::slices] for j in range(slices)] for i in range(n)]
    autoencoders = [BetaVAE_H(z_dim=1, seq_length=len(M_slices[0][i])).to(args.device) for i in range(slices)]
    # optimizers = [torch.optim.Adam(autoencoder.parameters(), lr=0.001) for autoencoder in autoencoders]
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for i, autoencoder in enumerate(autoencoders):
        autoencoder.load_state_dict(torch.load('./save/FUL/{}/N{}/E{}/VAE{}_{}.pth'.format(args.dataset, args.num_users,args.epochs,args.AE_epochs,i),map_location=args.device))
    M_last = M_slices[-1]
    M_test = copy.deepcopy(M[-1])
    net_glob = vectomodel_vit(M_test,net_glob)
    acc,loss = test_vit(net_glob, dataset_test, args)
    print('acc:',acc,'loss:',loss)

    for epoch in range(1):
        for j in range(slices):
            output, mu, logvar = autoencoders[j](M_last[j])  # loss等于所有的欧氏距离
            M_last[j] = output.detach().clone()
            # 测试cos
            print(cos(M_last[j], MU_slices[-1][j]))
    # 将M_last 拼回M_test
    M_test_new = M_test.clone().detach()
    for j in range(slices):
        M_test_new[j::slices] = M_last[j].detach().clone()
    # 计算相似度
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    print(cos(M_test_new, MU[-1]))

    net_glob = vectomodel_vit(M_test_new,net_glob)
    acc,loss = test_vit(net_glob, dataset_test, args)
    print('new_acc:',acc,'new_loss:',loss)
    # 用mnist数据集本地训练
    val_acc_list = []
    val_loss_list = []
    val_loss_list.append(loss)
    val_acc_list.append(acc)

    # 定位
    for i in range(args.epochs):
        if acc <= fl_acc[i]:
            pro_start = i
            break
    copy_args = copy.deepcopy(args)
    copy_args.lr = learning_rate
    print('pro_start:',pro_start)
    dict_users = mnist_iid(dataset_train, 10)
    start_time = time.time()
    for epoch in range(args.post_epochs):
        local = LocalUpdate_vit(args=copy_args, dataset=dataset_train, idxs=dict_users[0])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        # 更新全局模型
        # 取参数转为一维参数
        iter = epoch + pro_start
        if  iter >= args.epochs - 1:
            iter = args.epochs - 1
        former = torch.cat([param.view(-1) for name, param in net_glob.classifier.named_parameters() if 'lora' in name])
        net_glob.load_state_dict(w)
        latter = torch.cat([param.view(-1) for name, param in net_glob.classifier.named_parameters() if 'lora' in name])
        delta = latter - former
        delta_M = M[iter] - M[iter - 1]
        # 将向量M投影到向量U上
        new_U = torch.norm(delta_M) * delta / torch.norm(delta)
        tmp2 = former + new_U
        net_glob = vectomodel_vit(tmp2, net_glob)
        acc, loss = test_vit(net_glob, dataset_test, args)
        val_acc_list.append(acc)
        val_loss_list.append(loss)

    end = time.time()
    print('time:', end - start_time)
    # 绘制图像
    # 加载val_acc.txt 和val_loss.txt

    with open('./save/FL/{}/{}/N{}/E{}/val_acc.txt'.format(args.dataset, args.lr,args.num_users,args.epochs), 'r') as f:
        val_acc = [float(i) for i in f.readlines()]
    with open('.save/FL/{}/{}/N{}/E{}/val_loss.txt'.format(args.dataset,args.lr, args.num_users,args.epochs), 'r') as f:
        val_loss = [float(i) for i in f.readlines()]

    with open('./save/FUL/{}/N{}/E{}/compare/AE{}acc.txt'.format(args.dataset, args.num_users, args.epochs,args.AE_epochs), 'w') as f:
        for acc in val_acc_list:
            f.write(str(acc) + '\n')
    with open('./save/FUL/{}/N{}/E{}/compare/AE{}loss.txt'.format(args.dataset, args.num_users, args.epochs,args.AE_epochs), 'w') as f:
        for loss in val_loss_list:
            f.write(str(loss) + '\n')
    with open('./save/FUL/{}/N{}/E{}/compare/FLacc.txt'.format(args.dataset, args.num_users, args.epochs), 'w') as f:
        for acc in val_acc:
            f.write(str(acc) + '\n')
    with open('./save/FUL/{}/N{}/E{}/compare/FLloss.txt'.format(args.dataset, args.num_users, args.epochs), 'w') as f:
        for loss in val_loss:
            f.write(str(loss) + '\n')
    # 保存时间
    with open('./save/FUL/{}/N{}/E{}/time.txt'.format(args.dataset, args.num_users,args.epochs), 'w') as f:
        f.write(str(end - start_time) + '\n')
    plt.figure()
    plt.plot(range(args.post_epochs + 1), val_acc_list, label='DPUL')
    plt.plot(range(len(val_acc)), val_acc, label='FL')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Training Rounds')
    # xtick为整数
    plt.xticks(np.arange(0, args.post_epochs + 1, 1))
    plt.legend()
    # 如果没有文件夹则创建文件夹

    plt.savefig(
        './save/FUL/{}/N{}/E{}/compare/VAE{}acc.png'.format(args.dataset, args.num_users, args.epochs, args.AE_epochs))
    plt.figure()
    plt.plot(range(args.post_epochs + 1), val_loss_list, label='DPUL')
    plt.plot(range(len(val_loss)), val_loss, label='FL')
    plt.ylabel('Test Loss')
    plt.xlabel('Training Rounds')
    plt.xticks(np.arange(0, args.post_epochs +1, 1))
    plt.legend()
    plt.savefig(
        './save/FUL/{}/N{}/E{}/compare/VAE{}loss.png'.format(args.dataset, args.num_users, args.epochs, args.AE_epochs))
    # 保存准确率和损失值以text形式及


