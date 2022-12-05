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

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--temp', type=float)

    args = parser.parse_args()

    return args


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

class ResNet20Encoder(nn.Module):
    def __init__(self, num_layers=20, num_stem_conv=16, config=(16, 32, 64)):
        super(ResNet20Encoder, self).__init__()
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
        return features

    
class SimCLR(nn.Module):
    def __init__(self, num_layers=20, num_stem_conv=16, config=(16, 32, 64), projection_dim=20):
        super(SimCLR, self).__init__()
        self.encoder = ResNet20Encoder(num_layers=num_layers, num_stem_conv=num_stem_conv, config=config)
        self.linear1 = nn.Linear(config[-1], config[-1], bias=False)
        self.linear2 = nn.Linear(config[-1], projection_dim, bias=False)
        
    def forward(self, aug1, aug2):
        aug1_out = self.encoder(aug1).squeeze()
        aug1_out = self.linear1(aug1_out)
        aug1_out = nn.ReLU()(aug1_out)
        aug1_out = self.linear2(aug1_out)
        
        aug2_out = self.encoder(aug2).squeeze()
        aug2_out = self.linear1(aug2_out)
        aug2_out = nn.ReLU()(aug2_out)
        aug2_out = self.linear2(aug2_out)
        
        return aug1_out, aug2_out



class SimCLRTransform:
    
    def __init__(self):
        
        color_jitter = torchvision.transforms.ColorJitter(
            0.4, 0.4, 0.4, 0.1
        )
        self.simclr_transform = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(32, 32)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.simclr_transform(x), self.simclr_transform(x)


def nt_xent(aug1_out, aug2_out, temperature=1.0):
    batch_size = aug1_out.shape[0]
    augs = torch.cat((aug1_out, aug2_out), dim=0)
    cos_sim = torch.matmul(torch.nn.functional.normalize(augs, dim=1) , torch.nn.functional.normalize(augs, dim=1).T)
    cos_sim /= temperature
    
    cos_sim_diag = torch.diag(cos_sim)
    
    loss = None
    for i in range(2 * batch_size):
        if i < batch_size:
            positive_idx = i + batch_size
        else:
            positive_idx = i - batch_size
        
        row_loss = -torch.log(torch.exp(cos_sim[i][positive_idx]) / 
                              (torch.sum(torch.exp(cos_sim[i])) - torch.exp(cos_sim_diag[i])))

        if loss == None:
            loss = row_loss
        else:
            loss += row_loss
    loss /= (2 * batch_size)
    return loss
    
    


if __name__ == '__main__':

    print('running', flush=True)

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    TEMP = args.temp

    print(args, BATCH_SIZE, EPOCHS, LR, TEMP, flush=True)

    train_transform = SimCLRTransform()
    test_transform = transforms.Compose(
        [transforms.Resize(size=(32, 32)),
        transforms.ToTensor()]
    )

    trainset = torchvision.datasets.CIFAR10(root='/usr/xtmp/zg78/cifar10_data/', train=True, download=True, transform=train_transform)
    # trainset, valset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), int(len(trainset)*0.2)])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # valoader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='/usr/xtmp/zg78/cifar10_data/', train=False, download=True, transform=train_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_transform = SimCLRTransform()
    test_transform = transforms.Compose(
        [transforms.Resize(size=(32, 32)),
        transforms.ToTensor()]
    )

    simclr_model = SimCLR(projection_dim=64).cuda()
    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=LR)

    best_loss = 9999999

    for epoch_idx in range(EPOCHS):
        epoch_losses = 0
        simclr_model.train()
        for batch_idx, data in enumerate(trainloader):
            image, _ = data
            
            simclr_model.zero_grad()
            aug1, aug2 = image
            aug1 = aug1.cuda()
            aug2 = aug2.cuda()
            aug1_out, aug2_out = simclr_model(aug1, aug2)
            loss = nt_xent(aug1_out, aug2_out, temperature=TEMP)
            loss.backward()
            optimizer.step()

            epoch_losses += loss
        
        epoch_losses /= len(trainloader)
    
        with torch.no_grad():
            simclr_model.eval()
            val_epoch_losses = 0
            for batch_idx, data in enumerate(testloader):
                image, _ = data
                
                aug1, aug2 = image
                aug1 = aug1.cuda()
                aug2 = aug2.cuda()
                aug1_out, aug2_out = simclr_model(aug1, aug2)
                loss = nt_xent(aug1_out, aug2_out, temperature=TEMP)
                val_epoch_losses += loss
            val_epoch_losses /= len(testloader)

        save_name = epoch_idx // 100

        if val_epoch_losses < best_loss:
            best_loss = val_epoch_losses
            torch.save(simclr_model.state_dict(), f'models/simclr_pretrained/simclr_model_valwtest_{save_name}*100_{EPOCHS}_{BATCH_SIZE}_{TEMP}_{LR}.pth')


        print(f'{epoch_idx} | Train Loss {epoch_losses} ; Test Loss {val_epoch_losses}', flush=True)