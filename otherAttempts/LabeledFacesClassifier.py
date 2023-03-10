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

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(lfw_people)

transformer = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
])

lfw_train = datasets.LFWPeople('.', download=True, split="train", transform=transformer)
lfw_test = datasets.LFWPeople('.', download=True, split="test", transform=transformer)

# Dataset LFWPeople
#     Number of datapoints: 9525
#     Root location: .\lfw-py
#     Alignment: funneled
#     Split: train
#     Classes (identities): 5749
#     StandardTransform
# Transform: Compose(
#                ToTensor()
#            )
# Dataset LFWPeople
#     Number of datapoints: 3708
#     Root location: .\lfw-py
#     Alignment: funneled
#     Split: test
#     Classes (identities): 5749
#     StandardTransform
# Transform: Compose(
#                ToTensor()
#            )
print(lfw_train)
print(lfw_test)

#train_data, test_data = random_split(lfw_dataset, [10586, 2647])
BATCH_SIZE = 100

trainloader = DataLoader(lfw_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(lfw_test, batch_size=BATCH_SIZE)
for epoch in range(2):
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = np.array(inputs)
        labels = np.array(labels)
        print(inputs.shape)
        print(labels.shape)
        break
    break
# ---------------------------------------

import os
#from deepface.commons import functions

# ---------------------------------------


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.convLayerSet1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.convLayerSet2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.linear1 = nn.Linear(in_features=238144, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=5749)
        # self.convLayer1 = nn.Sequential(
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(3, 64, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(64, 64, (3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2), stride=(2, 2))
        # )

        # self.convLayer2 = nn.Sequential(
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(64, 128, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(128, 128, (3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2), stride=(2, 2)) 
        # )

        # self.convLayer3 = nn.Sequential(
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(128, 256, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(256, 256, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(256, 256, (3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2), stride=(2, 2))
        # )

        # self.convLayer4 = nn.Sequential(
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(256, 512, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2), stride=(2, 2))
        # )

        # self.convLayer5 = nn.Sequential(
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d((1, 1)),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2), stride=(2, 2))
        # )

        # self.convLayer6 = nn.Sequential(
        #     nn.Conv2d(512, 4096, (2, 7)),
        #     nn.Dropout(p=0.5),
        #     nn.Conv2d(4096, 4096, (1, 1)),
        #     nn.Dropout(p=0.5),
        #     nn.Conv2d(4096, 2622, (1, 1)),
        #     nn.Flatten(),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        x = self.convLayerSet1(x)
        x = self.convLayerSet2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.log_softmax(x)
        return x

        return x
    

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
# TODO: Define loss function 
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
num_epoch = 5 # TODO: Choose an appropriate number of training epochs

def train(model, train_loader, num_epoch = 5): # Train the model
    training_loss = []
    #validation_loss = []
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        training_running_loss = []
        for batch, label in tqdm(train_loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            training_running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(training_running_loss))) # Print the average loss for this epoch
        training_loss.append(np.mean(training_running_loss))

        #calculate the validation loss by setting model to evaluation mode
        # validation_accuracy, val_loss = evaluate(model, val_loader)
        # validation_loss.append(val_loss)
    print("Done!")
    return training_loss


def evaluate(model, loader): # Evaluate accuracy on validation / test set
    running_loss = []
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc, np.mean(running_loss)

#run training and validation for training. 
training_loss = train(model, trainloader, 5)
#save the model
torch.save(model.state_dict(), "bestModel.pt")

print("Evaluate on test set")
evaluate(model, testloader)

# #plot the validation and training loss
# epochs = [1, 2, 3, 4, 5]
# plt.plot(epochs, training_loss, label='train_loss')
# plt.plot(epochs, validation_loss, label='val_loss')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# define loss function, optimizer and number of epochs
# train and test to get baseline accuracy


# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd
# Pick attacks here, easy to execute them
