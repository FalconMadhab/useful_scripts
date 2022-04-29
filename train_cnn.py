import os
import math
import numpy as np
import torch
from torchvision import transforms,models,datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from torch.utils.data.sampler import SubsetRandomSampler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)
from time import time


from torchsummary import summary

data_transf = transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomAutocontrast(p=0.5),
                                  transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                  transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                  transforms.Resize((256,256)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean =  [0.485, 0.456, 0.406],
                                                       std  =  [0.229, 0.224, 0.225])])

# data_transf = transforms.Compose([transforms.Resize((224,224)),
#                                   transforms.ToTensor()])

train_data = datasets.ImageFolder(root= '/home/ninad/fsm_classifier', transform = data_transf)

target = train_data.targets

train_idx, valid_idx= train_test_split(
np.arange(len(target)),
test_size=0.2,
shuffle=True,
stratify=target)

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders

trainloader = DataLoader(dataset = train_data, batch_size=128, drop_last=True,sampler=train_sampler)
validloader = DataLoader(dataset = train_data, batch_size=128,drop_last=True,sampler=valid_sampler)

print(len(train_sampler))
print(len(valid_sampler))
label2 = list(train_data.class_to_idx.keys())
images,labels=next(iter(trainloader))
print("Images shape after loading data in dataloader: ",images.shape)
print("labels: ",label2)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2,2),

        ).to(device)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(4096, 256),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        ).to(device)
        
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


model = MyModel().to(device)
summary(model, (3,256,256))

def Train(epoch, print_every=1):
    total_loss = 0
    start_time = time()
    
    accuracy = []
    
    for i, batch in enumerate(trainloader, 1):
        minput = batch[0].to(device) # Get batch of images from our train dataloader
        target = batch[1].to(device) # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas
        
        moutput = model(minput) # output by our model
        
        loss = criterion(moutput, target) # compute cross entropy loss
        total_loss += loss.item()

        optimizer.zero_grad() # Clear the gradients if exists. (Gradients are used for back-propogation.)
        loss.backward() # Back propogate the losses
        optimizer.step() # Update Model parameters
        
        argmax = moutput.argmax(dim=1) # Get the class index with maximum probability predicted by the model
        accuracy.append((target==argmax).sum().item() / target.shape[0]) # calculate accuracy by comparing to target tensor

        if i%print_every == 0:
            print('Epoch: [{}]/({}/{}), Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
                epoch, i, len(trainloader), loss.item(), sum(accuracy)/len(accuracy), time()-start_time 
            ))
    
    return total_loss / len(trainloader) # Returning Average Training Loss

lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
train_loss = []
for epoch in range(1, 51):
    train_loss.append(Train(epoch,2))
    # test_loss.append(Test(epoch))

    print('\n')
    
    if epoch % 10 == 0:
        torch.save(model, 'model_'+str(epoch)+'.pth')