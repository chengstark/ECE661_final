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
    parser.add_argument('--label_percentage', type=float, default=0.1)

    args = parser.parse_args()
    return args


class DirectFwd(nn.Module):
    def __init__(self):
        super(DirectFwd, self).__init__()

    def forward(self, x):
        return x


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = DirectFwd()
        self.model.fc = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        out = self.model(x)
        return out

    
if __name__ == '__main__':

    print('running', flush=True)

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    label_percentage = args.label_percentage
    print('BASELINEBASELINE', flush=True)
    print(args, BATCH_SIZE, EPOCHS, LR, label_percentage, flush=True)

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
    trainset, _ = torch.utils.data.random_split(trainset, [int(len(trainset)*label_percentage), int(len(trainset)*(1-label_percentage))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='/usr/xtmp/zg78/cifar10_data/', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print('data finished', flush=True)

    criterion = nn.CrossEntropyLoss()

    model = BaselineModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = 9999999

    for epoch_idx in range(EPOCHS):
        epoch_losses = 0
        epoch_correts = 0
        model.train()
        for batch_idx, data in enumerate(trainloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            
            model.zero_grad()
            out = model(image)
            
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            epoch_losses += loss
        
            pred = torch.argmax(out, dim=1)
            epoch_correts += torch.sum(pred == label).item()
        
        epoch_losses /= len(trainloader)
        epoch_correts /= len(trainset)
            
        
        with torch.no_grad():
            model.eval()
            
            val_epoch_losses = 0
            val_epoch_correts = 0

            for batch_idx, data in enumerate(testloader):
                image, label = data
                image = image.cuda()
                label = label.cuda()

                out = model(image)

                loss = criterion(out, label)

                val_epoch_losses += loss

                pred = torch.argmax(out, dim=1)
                val_epoch_correts += torch.sum(pred == label).item()

            val_epoch_losses /= len(testloader)
            val_epoch_correts /= len(testset)

            if val_epoch_losses < best_loss:
                best_loss = val_epoch_losses
                torch.save(model.state_dict(), f'models/supervised_baseline/{EPOCHS}_{BATCH_SIZE}_{LR}_{label_percentage}.pth')
            
        print(f'{epoch_idx} | Train Loss {epoch_losses} Acc {epoch_correts} ; Val Loss {val_epoch_losses} Acc {val_epoch_correts}', flush=True)


