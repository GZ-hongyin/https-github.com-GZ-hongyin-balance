import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm
from model import resnet32
from dataset import MnistSubset
from losses import FocalLoss,LDAMLoss

import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Lambda(lambda x: x.repeat(3,1,1)),
     transforms.Normalize((0.5,),(0.5,))]  #image = (image-mean)-std
)

#trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers = 0)

selnumber = np.linspace(6000,60,10,dtype=int)  
trainset = MnistSubset(selnum=selnumber,mode='train')
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers = 0)

testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers = 0)

cls_num_list = trainset.get_cls_num_list()

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3),#(28-3+2*0)/1+1,26*26*16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),#(26-3+2*0)/1+1=24,24*24*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)# (24-2+2*0)/2+1,12*12*32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), # (12-3+2*0)/1+1, 10*10*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),#,(10-3+2*0)/1+1,8*8*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) #(8-2+2*0)/2+1,4*4*128
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN4()
if torch.cuda.is_available():
    model = model.cuda()



criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss(gamma=2,weight=None)
#criterion = LDAMLoss(cls_num_list = cls_num_list,max_m=0.5,s=30,weight=None)
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)


for epoch in range(2): 
    running_loss = 0.0
    for i ,data in enumerate(tqdm(trainloader),0):
       
        inputs,labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        #zero the parameter gradients
        optimizer.zero_grad()

        # forward backward optimize
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        
        running_loss += loss.item()
        if i % 2000 ==1999:
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print('train result ')


#device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#print(device)
#CNN4.to(device)
#inputs,labels = data[0].to(device),data[1].to(device)


model.eval()
eval_loss = 0
eval_acc = 0
with torch.no_grad():
 for data in testloader:
    inputs,labels = data
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    outputs = model(inputs)
    loss = criterion(outputs,labels)
    eval_loss += loss.data.item()*labels.size(0)
    _, pred = torch.max(outputs,1)
    num_correct = (pred==labels).sum()
    eval_acc += num_correct.item()
 print('Acc:{:.6f}'.format(eval_acc/(len(testset))))

