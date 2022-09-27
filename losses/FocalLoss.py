import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


if __name__ == '__main__':
    from utils import fix_random
    fix_random(seed=0)
    input = torch.rand(5,10) #（batch数，类别数）
    target = torch.randint(10,size=(5,))

    criterion = FocalLoss(gamma=2,weight=None)
    print('criterion: ',criterion)
    loss = criterion(input,target)
    print('loss: ',loss)
