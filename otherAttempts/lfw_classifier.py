# -*- coding: utf-8 -*-
"""anotherTest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hU2jtViZK3L-LxunDXcJb46H6-bujpv9
"""

import numpy as np
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing 
from sklearn.datasets import fetch_lfw_people
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import skorch
import torchvision

#! pip install skorch

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

data = lfw_people.data
size = data.shape[1]
label = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0] 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_samples, h, w = lfw_people.images.shape

print("Total dataset size:")
print("n_features: %d" % size)
print("n_classes: %d" % n_classes)

X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.25, random_state=1, stratify=y)

class Resnet18(nn.Module):
    """ResNet 18, pretrained, with one input chanel and 7 outputs."""
    def __init__(self):
        super(Resnet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 7)
    def forward(self, x):
        return self.model(x)

torch.manual_seed(0)
resnet = NeuralNetClassifier(
    Resnet18,
    criterion=nn.CrossEntropyLoss,
    max_epochs=50,
    batch_size=128, 
    optimizer=torch.optim.Adam,
    optimizer__lr=0.001,
    optimizer__betas=(0.9, 0.999),
    optimizer__eps=1e-4,
    optimizer__weight_decay=0.0001,  
    train_split=skorch.dataset.CVSplit(cv=5, stratified=True),
    device=device,
    verbose=0)

scaler = preprocessing.MinMaxScaler()
X_train_s = scaler.fit_transform(X_train).reshape(-1, 1, h, w)
X_test_s = scaler.transform(X_test).reshape(-1, 1, h, w)

t0 = time()
resnet.fit(X_train_s, y_train)
print("done in %0.3fs" % (time() - t0))

"""Continue training a model (warm re-start):<br>
resnet.partial_fit(X_train_s, y_train)
"""

y_pred = resnet.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=target_names))