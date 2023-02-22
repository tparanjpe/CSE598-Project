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
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import tensorflow as tf


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
#print(lfw_people)
print(lfw_people.target.shape)
print(lfw_people.data.shape)
print(type(lfw_people))
# for name in lfw_people.target:
#     print(name)
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size = .2, shuffle=True)
print(len(X_test))
print(len(y_test))
print(len(X_train))
print(len(y_train))


tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Dense,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Dense,
    )

# def baseModel():
#     model = Sequential()
#     model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
#     model.add(Convolution2D(64, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(64, (3, 3), activation="relu"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(128, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(128, (3, 3), activation="relu"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(256, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(256, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(256, (3, 3), activation="relu"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, (3, 3), activation="relu"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, (3, 3), activation="relu"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     model.add(Convolution2D(4096, (7, 7), activation="relu"))
#     model.add(Dropout(0.5))
#     model.add(Convolution2D(4096, (1, 1), activation="relu"))
#     model.add(Dropout(0.5))
#     model.add(Convolution2D(2622, (1, 1)))
#     model.add(Flatten())
#     model.add(Activation("softmax"))

#     return model

def baseModel():
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape = (250, 250, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size =(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


model = baseModel()
history = model.fit(X_train, y_train, batch_size=100, epochs=2)
print(history)#