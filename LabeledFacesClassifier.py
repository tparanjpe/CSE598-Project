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

# ---------------------------------------

import os
import gdown
from deepface.commons import functions

# ---------------------------------------


def baseModel():
    model = nn.Sequential(
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(512, 4096, (7, 7)),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, (1, 1)),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 2622, (1, 1)),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    return model


# url = 'https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo'


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
):

    model = baseModel()

    # -----------------------------------

    home = functions.get_deepface_home()
    output = home + "/.deepface/weights/vgg_face_weights.h5"

    if os.path.isfile(output) != True:
        print("vgg_face_weights.h5 will be downloaded...")
        gdown.download(url, output, quiet=False)

    # -----------------------------------

    model.load_weights(output)

    # -----------------------------------

    # TO-DO: why?
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor
    
# define loss function, optimizer and number of epochs
# train and test to get baseline accuracy


# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd
# Pick attacks here, easy to execute them
