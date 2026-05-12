import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np 
import pandas as pd 
from torch.optim import Adam
import matplotlib.pyplot as plt 
from google.colab import files
from PIL import Image

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          
    transforms.RandomHorizontalFlip(),             
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),    
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset=datasets.CIFAR10(root='./data',train=True, download=False, transform=transform)
test_dataset=datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader=DataLoader(train_dataset,batch_size=64, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False)
data=[]
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,1,1)
        self.norm1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,3,1,1)
        self.norm2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,128,3,1,1)
        self.norm3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,512,3,1,1)
        self.norm4=nn.BatchNorm2d(512)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.3)

        self.pool=nn.MaxPool2d(2,2)
        
        self.fc1=nn.Linear(512*2*2,1024)
        self.fc2=nn.Linear(1024,10)

    def forward(self,x):
        x=self.pool(self.relu(self.norm1(self.conv1(x))))

        #block2
        x=self.pool(self.relu(self.norm2(self.conv2(x))))
        
        #block3
        x=self.pool(self.relu(self.norm3(self.conv3(x))))

        #block4
        x=self.pool(self.relu(self.norm4(self.conv4(x))))


        x=x.view(x.size(0),-1)

        x=self.dropout(x)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
model=CNN().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

loss_function=nn.CrossEntropyLoss()
for epoch in range(1,41):
    model.train()
    for images, labels in train_loader:
        images,labels=images.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction=model(images)
        check_loss=loss_function(prediction, labels)
        check_loss.backward()
        optimizer.step()
    scheduler.step()
    data.append(check_loss.item())
    print(f"Epoch no: {epoch} ---------> loss{check_loss.item()}")

model.eval()
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels= images.to(device), labels.to(device)
        output=model(images)
        _, prediction=torch.max(output,1)
        total=total+labels.size(0)
        correct=correct+(prediction==labels).sum().item()
accuracy=100*correct/total
print("The over all accuracy is: ", accuracy)
plt.plot(data,marker='o')

plt.show()
torch.save(model.state.dict(),"cifar_model.pth")
files.download("cifar_model.pth")