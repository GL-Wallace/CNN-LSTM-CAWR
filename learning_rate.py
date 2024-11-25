import torch.optim as optim
import torch
import torch.nn as nn
import argparse
import math
from copy import copy
import matplotlib.pyplot as plt

class CosineAnnealingWarmbootingLR:
    # cawb learning rate scheduler: given the warm booting steps, calculate the learning rate automatically

    def __init__(self, optimizer, epochs=0, eta_min=0.05, steps=[], step_scale=0.8, lf=None, batchs=0, warmup_epoch=20,
                 epoch_scale=1.0):
        self.warmup_iters = batchs * warmup_epoch
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.iters = -1
        self.iters_batch = -1
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.step_scale = step_scale
        steps.sort()
        self.steps = [warmup_epoch] + [i for i in steps if (i < epochs and i > warmup_epoch)] + [epochs]
        self.gap = 0
        self.last_epoch = 0
        self.lf = lf
        self.epoch_scale = epoch_scale

        # Initialize epochs and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def step(self, external_iter=None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter

        # cos warm boot policy
        iters = self.iters + self.last_epoch
        scale = 1.0
        for i in range(len(self.steps) - 1):
            if (iters <= self.steps[i + 1]):
                self.gap = self.steps[i + 1] - self.steps[i]
                iters = iters - self.steps[i]

                if i != len(self.steps) - 2:
                    self.gap += self.epoch_scale
                break
            scale *= self.step_scale

        if self.lf is None:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr * ((((1 + math.cos(iters * math.pi / self.gap)) / 2) ** 1.0) * (
                            1.0 - self.eta_min) + self.eta_min)
        else:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr * self.lf(iters, self.gap)

        return self.optimizer.param_groups[0]['lr']

    def step_batch(self):
        self.iters_batch += 1

        if self.iters_batch < self.warmup_iters:

            rate = self.iters_batch / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return self.optimizer.param_groups[0]['lr']
        else:
            return None


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir='./LR.png'):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(scheduler.last_epoch):
        y.append(None)
    for _ in range(scheduler.last_epoch, epochs):
        y.append(scheduler.step())

    plt.plot(y, '.-', label='LR')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=800)
    plt.show()