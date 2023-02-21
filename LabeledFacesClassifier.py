import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.autograd import Variable
import random
import torchattacks

lfw_dataset = datasets.LFWPeople('.', download=True)

# Dataset LFWPeople
#     Number of datapoints: 13233
#     Root location: .\lfw-py
#     Alignment: funneled
#     Split: 10fold
#     Classes (identities): 5749

train_data, test_data = random_split(lfw_dataset, [10586, 2647])

# Needs to be updated!!!
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # This is the first Conv layer (sequential with batch norm, max pool, and reLU)
        self.convLayerSet1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
                
        # This is the second Conv layer (sequential with batch norm, max pool, and reLU)
        self.convLayerSet2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        #Linear layers
        self.linear1 = nn.Linear(in_features=1600, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
        x = self.convLayerSet1(x)
        x = self.convLayerSet2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.log_softmax(x)
        return x
    
# define loss function, optimizer and number of epochs
# train and test to get baseline accuracy


# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd
# Pick attacks here, easy to execute them
