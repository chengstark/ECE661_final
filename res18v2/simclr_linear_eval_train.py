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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained_path', type=str)
    args = parser.parse_args()
    return args

    
class DirectFwd(nn.Module):
    def __init__(self):
        super(DirectFwd, self).__init__()

    def forward(self, x):
        return x

    
class SimCLR(nn.Module):
    def __init__(self, projection_dim):
        super(SimCLR, self).__init__()

        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = DirectFwd()
        self.encoder.fc = DirectFwd()

        self.linear1 = nn.Linear(512, 512, bias=False)
        self.linear2 = nn.Linear(512, projection_dim, bias=False)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        
    def forward(self, aug1, aug2):
        aug1_out = self.encoder(aug1)
        aug1_out = self.linear1(aug1_out)
        aug1_out = self.bn1(aug1_out)
        aug1_out = nn.ReLU()(aug1_out)
        aug1_out = self.linear2(aug1_out)
        
        aug2_out = self.encoder(aug2)
        aug2_out = self.linear1(aug2_out)
        aug2_out = self.bn2(aug2_out)
        aug2_out = nn.ReLU()(aug2_out)
        aug2_out = self.linear2(aug2_out)
        
        return aug1_out, aug2_out
    
    
class SimCLR_linear_eval(nn.Module):
    def __init__(self, encoder):
        super(SimCLR_linear_eval, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(512, 10, bias=False)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.linear(out)
        return out

    
if __name__ == '__main__':

    print('running', flush=True)

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    pretrained_path = args.pretrained_path

    print(args, BATCH_SIZE, EPOCHS, LR, pretrained_path, flush=True)

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

    print('data finished', flush=True)

    criterion = nn.CrossEntropyLoss()

    simclr_model = SimCLR(projection_dim=128)
    simclr_model.load_state_dict(torch.load(pretrained_path))
    simclr_model.cuda()

    print('encoder loaded', flush=True)

    simclr_linear_eval_model = SimCLR_linear_eval(simclr_model.encoder)
    simclr_linear_eval_model.cuda()

    optimizer = torch.optim.Adam(simclr_linear_eval_model.parameters(), lr=LR)

    best_loss = 9999999

    for epoch_idx in range(EPOCHS):
        epoch_losses = 0
        epoch_correts = 0
        simclr_linear_eval_model.train()
        for batch_idx, data in enumerate(trainloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            
            simclr_linear_eval_model.zero_grad()
            out = simclr_linear_eval_model(image)
            
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            epoch_losses += loss
        
            pred = torch.argmax(out, dim=1)
            epoch_correts += torch.sum(pred == label).item()
        
        epoch_losses /= len(trainloader)
        epoch_correts /= len(trainset)
            
        
        with torch.no_grad():
            simclr_linear_eval_model.eval()
            
            val_epoch_losses = 0
            val_epoch_correts = 0

            for batch_idx, data in enumerate(testloader):
                image, label = data
                image = image.cuda()
                label = label.cuda()

                out = simclr_linear_eval_model(image)

                loss = criterion(out, label)

                val_epoch_losses += loss

                pred = torch.argmax(out, dim=1)
                val_epoch_correts += torch.sum(pred == label).item()

            val_epoch_losses /= len(testloader)
            val_epoch_correts /= len(testset)

            if val_epoch_losses < best_loss:
                best_loss = val_epoch_losses
                torch.save(simclr_linear_eval_model.state_dict(), f'models/simclr_linear_eval/simclr_model_linear_eval_{EPOCHS}_{BATCH_SIZE}_{LR}_{pretrained_path.split("/")[-1][:-4]}.pth')
            
        print(f'{epoch_idx} | Train Loss {epoch_losses} Acc {epoch_correts} ; Val Loss {val_epoch_losses} Acc {val_epoch_correts}', flush=True)


