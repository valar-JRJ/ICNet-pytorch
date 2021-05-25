"""Popular Learning Rate Schedulers"""
from __future__ import division
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

from bisect import bisect_right

__all__ = ['IterationPolyLR', 'ConstantLR']


class IterationPolyLR(_LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, last_epoch=-1):
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        super(IterationPolyLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        N = self.max_iters 
        T = self.last_epoch
        factor = pow(1 - T / N, self.power)
        # https://blog.csdn.net/mieleizhi0522/article/details/83113824
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PloyStepLR(_LRScheduler):
    def __init__(self, optimizer, milestone=8000, gamma=0.5, last_epoch=-1):
        self.milestone = milestone
        self.gamma = gamma
        super(PloyStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = self.gamma**(self.last_epoch/self.milestone)
        return [base_lr*factor for base_lr in self.base_lrs]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    gamma = 0.5
    milestone = 8000
    max_iter = 66150
    init_lr = 0.01
    x = np.arange(0, max_iter)
    y_step = init_lr*gamma**(x/milestone)
    plt.title('polystep')
    plt.plot(x, y_step)
    plt.show()
