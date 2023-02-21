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

lfw_dataset = datasets.LFWPeople('.', download=True)

# Dataset LFWPeople
#     Number of datapoints: 13233
#     Root location: .\lfw-py
#     Alignment: funneled
#     Split: 10fold
#     Classes (identities): 5749

train_data, test_data = random_split(lfw_dataset, [10586, 2647])
