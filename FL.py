import copy
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torchvision import datasets
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification,ViTModel

from models.Autoencoder import AutoEncoder
from models.Fed import FedAvg
from models.FedAvg import FedAvg_noniid_float, FedAvg_noniid_float_www
from models.Nets import CNNMnist, CNNCifar, MLP, LeNet5, VGG11
from models.Update import LocalUpdate, LocalUpdate_vit
from models.Vectomodel import vectomodel, vectomodel_vit
from models.seed import setup_seed
from models.test import test_img, test_vit
from utils.options import args_parser
from utils.sample import cifar100_noniid, imagenet_tiny_noniid, cinic10_noniid, cifar_noniid
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.seed import set_seed
from utils.load_datasets import TinyImageNet, CINIC10

if __name__ == '__main__':
    # 初始化参数
    args = args_parser()
    args.lr = 5e-5
    args.epochs = 50
    args.local_ep = 5
    args.num_users = 10
    args.dataset = 'cifar10'
    args.model = 'Vits'
    args.iid = False
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if not os.path.exists('./save/FL/{}/{}/N{}/E{}'.format(args.dataset, args.lr, args.num_users, args.epochs)):
        os.makedirs('./save/FL/{}/{}/N{}/E{}'.format(args.dataset, args.lr, args.num_users, args.epochs))
    # load dataset
    if args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)
        dict_users = cifar_noniid(dataset_train, args.num_users)
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

    # load model 加载模型
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

    iter_new = 0
    # 查找储存的模型，查看到上次中断到第几轮，并加载模型，继续运行到最后
    for i in tqdm(range(args.epochs)):
        if os.path.exists('./save/FL/{}/{}/N{}/E{}/M{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,i)):
            M = torch.load('./save/FL/{}/{}/N{}/E{}/M{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,i), map_location=args.device)
            net_glob = vectomodel_vit(M, net_glob)
            iter_new = i + 1
        else:
            print('Load M{}.pth'.format(i - 1))
            iter_new = i
            break
        if os.path.exists('./save/FL/{}/{}/N{}/E{}/U{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,i)):
            U_w = torch.load('./save/FL/{}/{}/N{}/E{}/U{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,i), map_location=args.device)
            iter_new = i + 1
        else:
            print('Load U{}.pth'.format(i - 1))
            iter_new = i
            break
    # 测试准确率
    acc, loss = test_vit(net_glob, dataset_test, args)
    # 对epoch进行处理
    print('Epochs:{}'.format(iter_new))
    net_glob.train()
    net_test = copy.deepcopy(net_glob)
    # 复制全局模型
    w_glob = net_glob.state_dict()
    # 训练
    loss_train = []
    val_loss, val_acc = [], []
    val_loss.append(loss)
    val_acc.append(acc)
    client_data_sizes = [len(dict_users[i]) for i in range(args.num_users)]

    idxs_users = range(args.num_users)
    for iter in range(iter_new,args.epochs):
        loss_locals = []
        tmp1 = copy.deepcopy(net_glob.state_dict())
        tmp2 = copy.deepcopy(net_glob.state_dict())
        for idx in idxs_users:
            local = LocalUpdate_vit(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # aggragation in real time to save memory
            tmp1 = FedAvg_noniid_float_www(tmp1,w,tmp2,client_data_sizes,idx)
            loss_locals.append(copy.deepcopy(loss))
            # 存储遗忘客户端模型
            if idx == 0:
                net_test.load_state_dict(w)
                U_w = torch.cat([param.view(-1) for name, param in net_test.classifier.named_parameters() if'lora' in name])
                torch.save(U_w, './save/FL/{}/{}/N{}/E{}/U{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,iter))
        net_glob.load_state_dict(tmp1)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        # save parameter 将遍历w_glob的每一层参数转为向量格式存储
        M_glob = torch.cat([param.view(-1) for name, param in net_glob.classifier.named_parameters() if'lora' in name])
        # save acc 测试并保存准确率
        acc, loss = test_vit(net_glob, dataset_test, args)
        torch.save(M_glob, './save/FL/{}/{}/N{}/E{}/M{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,iter))

    # test acc 测试准确率
    for i in range(0,args.epochs):
        M = torch.load('./save/FL/{}/{}/N{}/E{}/M{}.pth'.format(args.dataset, args.lr,args.num_users, args.epochs,i), map_location=args.device)
        net_glob = vectomodel_vit(M, net_glob)
        acc, loss = test_vit(net_glob, dataset_test, args)
        val_acc.append(acc)
        val_loss.append(loss)
    # save acc list
    with open('./save/FL/{}/{}/N{}/E{}/val_acc.txt'.format(args.dataset, args.lr, args.num_users, args.epochs), 'w') as f:
        for acc in val_acc:
            f.write(str(acc) + '\n')
    with open('./save/FL/{}/{}/N{}/E{}/val_loss.txt'.format(args.dataset, args.lr, args.num_users, args.epochs), 'w') as f:
        for loss in val_loss:
            f.write(str(loss) + '\n')



