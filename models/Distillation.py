import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy

# 知识蒸馏
class Distill:
    def __init__(self, args) -> None:
        self.T = args.T
        pass
    def distillation_loss(self,outputs, teacher_outputs):
        criterion = nn.KLDivLoss(reduction='batchmean')
        Loss = criterion(nn.functional.log_softmax(outputs/self.T, dim=1),nn.functional.softmax(teacher_outputs/self.T, dim=1)) * (self.T*self.T)
        return   Loss
    def boost_loss(self,outputs, labels):
        criterion = nn.CrossEntropyLoss()
        Loss = criterion(outputs, labels)
        return  Loss
