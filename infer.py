# -*- coding: utf-8 -*-
"""car_model_predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HX2ULB9uTvIlLRVzrxrrX3PAoBcClAhp
"""

import os
import math
from statistics import mode
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
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

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


image=cv2.imread('/home/ninad/fsm_aug_data_classifier/train/FSM/4_1628483617.1348224_2.jpg')
print("Random image shape: ",image.shape)

data_transf = transforms.Compose([transforms.RandomHorizontalFlip(),
                                  transforms.Resize((224,224)),
                                  transforms.ToTensor()])

train_data = datasets.ImageFolder(root= '/home/ninad/fsm_aug_data_classifier/train', transform = data_transf)

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

trainloader = DataLoader(dataset = train_data, batch_size=32, drop_last=True,sampler=train_sampler)
validloader = DataLoader(dataset = train_data, batch_size=32,drop_last=True,sampler=valid_sampler)

print(len(train_sampler))
print(len(valid_sampler))

images,labels=next(iter(trainloader))
print("Images shape after loading data in dataloader: ",images.shape)

# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

# #freezing the initial layers of MobileNetv2
# for param in model.parameters():
#     param.requires_grad = False


# #adding our own classifier
model = MyModel().to(device)
# model.fc = nn.Sequential(
#                       nn.Linear(512, 128),
#                       nn.ReLU(), 
#                       nn.Dropout(0.2),
#                       nn.Linear(128, 64),
#                       nn.ReLU(),
#                       nn.Linear(64, 32),
#                       nn.ReLU(),
#                       nn.Dropout(0.2),
#                       nn.Linear(32, 2),
#                       nn.LogSoftmax(dim=1))    

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.kaiming_uniform(m.weight, nonlinearity='relu')
#         m.bias.data.fill_(0.01)

# model = model.apply(init_weights)


# model = model.to('cuda')

# model.load_state_dict(torch.load('model.pt'))
# model.to('cuda')
# print("Model",model)

'''
from torch.optim.lr_scheduler import ReduceLROnPlateau

criteria=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(), lr=3e-4)
# optimizer=optim.SGD(model.parameters(), lr = 3e-6, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.3, patience = 2, verbose = True)

epochs = 30
valid_loss_min = np.Inf
train_losses=[]
valid_losses=[]


for e in range(epochs):
    trainloss=0
    validloss=0
    
    
    model.train()
    for images,labels in trainloader:
        images,labels=images.to('cuda'),labels.to('cuda')
        optimizer.zero_grad()   
        out=model(images)
        loss=criteria(out,labels)
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
  
    with torch.no_grad():
      model.eval()
      num_correct=0
      num_examples=0
      for images,labels in validloader:
          images,labels=images.to('cuda'),labels.to('cuda')
          out=model(images)
          loss=criteria(out,labels)
          validloss+=loss.item()

          correct = torch.eq(torch.max(F.softmax(out,dim=1), dim=1)[1], labels).view(-1)
          num_correct += torch.sum(correct).item()
          num_examples += correct.shape[0]
          
      train_loss = trainloss/len(trainloader.sampler)
      train_losses.append(train_loss)

      valid_loss = validloss/len(validloader.sampler)
      valid_losses.append(valid_loss)

      scheduler.step(valid_loss)
    
      print(e+1)
      print('trainloss = '+str(train_loss)+'  validloss = '+str(valid_loss))
      print('valid_accuracy = '+str(num_correct/num_examples))

    
      if valid_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), 'model.pt')
          valid_loss_min = valid_loss
'''
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackened.figure_format = 'retina'

import matplotlib.pyplot as plt

# plt.plot(train_losses, label='trainloss')
# plt.plot(valid_losses, label='validloss')
# plt.legend(frameon = False)

# label for inference
label = list(train_data.class_to_idx.keys())
print(label)
# performing inference on an image
from PIL import Image
from torch.autograd import Variable

loader = transforms.Compose([transforms.Scale(224), transforms.ToTensor()
                             ])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU

image = image_loader('/home/ninad/fsm_aug_data_classifier/train/FSM/4_1628483617.1348224_2.jpg')

model = model.load_state_dict(torch.load('/home/ninad/fsm_model5.pt'))
# model.to('cuda')
# dummy_input = torch.randn(32, 3, 224, 224)

# torch.onnx.export(model, dummy_input, "fuelling_classifier.onnx")
model = Net(len(classes))
model.load_state_dict(torch.load(PATH))
model.eval()
out = model(image)
print("out",label[torch.max(F.softmax(out,dim=1), dim=1)[1]])
img = cv2.imread('/home/ninad/fsm_aug_data_classifier/train/FSM/4_1628483617.1348224_2.jpg')

plt.figure(figsize=(8, 8))

# give prediction of the label for the x-axis
plt.xlabel(label[torch.max(F.softmax(out,dim=1), dim=1)[1]], fontsize=18)
# plt.savefig('test1.png')
plt.imshow(img)
plt.show()

