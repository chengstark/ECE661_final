# import necessary dependencies
import argparse
import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm

import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-4


# train_transform = transforms.Compose(
#     [transforms.Resize(size=(32, 32)),
#     transforms.ToTensor()]
# )

color_jitter = torchvision.transforms.ColorJitter(
        0.4, 0.4, 0.4, 0.1
    )
train_transform = torchvision.transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
    transforms.ToTensor()]
)

trainset = torchvision.datasets.CIFAR10(root='/usr/xtmp/zg78/cifar10_data/', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='/usr/xtmp/zg78/cifar10_data/', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()


class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides):
        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                      stride=strides, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs, out_channels=out_chs,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs)
        )

        if in_chs != out_chs:
            self.id_mapping = nn.Sequential(
                nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                          stride=strides, padding=0, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chs))
        else:
            self.id_mapping = None
        self.final_activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None:
            x_ = self.id_mapping(x)
        else:
            x_ = x
        return self.final_activation(x_ + out)

class ResNet20base(nn.Module):
    def __init__(self, num_layers=20, num_stem_conv=16, config=(16, 32, 64)):
        super(ResNet20base, self).__init__()
        self.num_layers = num_layers
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_stem_conv,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_stem_conv),
            nn.ReLU(True)
        )
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = num_stem_conv
        for i in range(len(config)):
            for j in range(num_layers_per_stage):
                if j == 0 and i != 0:
                    strides = 2
                else:
                    strides = 1
                self.body_op.append(ResNet_Block(num_inputs, config[i], strides))
                num_inputs = config[i]
        self.body_op = nn.Sequential(*self.body_op)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(config[-1], 10)

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        features = self.avg_pool(out)
        out = self.final_fc(features.squeeze())
        return out


net = ResNet20base().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

best_loss = 9999999

for epoch_idx in range(EPOCHS):
    epoch_losses = 0
    epoch_correts = 0
    net.train()
    for batch_idx, data in enumerate(trainloader):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        
        net.zero_grad()
        out = net(image)
        
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        epoch_losses += loss
    
        pred = torch.argmax(out, dim=1)
        epoch_correts += torch.sum(pred == label).item()
    
    epoch_losses /= len(trainloader)
    epoch_correts /= len(trainset)
        
    
    with torch.no_grad():
        net.eval()
        
        val_epoch_losses = 0
        val_epoch_correts = 0

        for batch_idx, data in enumerate(testloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()

            out = net(image)

            loss = criterion(out, label)

            val_epoch_losses += loss

            pred = torch.argmax(out, dim=1)
            val_epoch_correts += torch.sum(pred == label).item()

        val_epoch_losses /= len(testloader)
        val_epoch_correts /= len(testset)

        if val_epoch_losses < best_loss:
            best_loss = val_epoch_losses
            torch.save(net.state_dict(), f'resent20base.pth')
    
    print(f'{epoch_idx} | Train Loss {epoch_losses} Acc {epoch_correts} ; Val Loss {val_epoch_losses} Acc {val_epoch_correts}', flush=True)