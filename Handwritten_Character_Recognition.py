import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.RandomRotation((-90,-90)),
    transforms.ToTensor()
])

train_data=datasets.EMNIST(root='./data',train=True,split='balanced',transform=transform,download=True)
test_data=datasets.EMNIST(root='./data',train=False,split='balanced',transform=transform,download=True)

train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000)

class EMNISTClassifier(nn.Module):
    def __init__(self):
        super(EMNISTClassifier,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.fc1=nn.Linear(64*7*7,256)
        self.fc2=nn.Linear(256,47)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)

model=EMNISTClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss.item():.4f} in epoch of {epoch}")

correct=0
total=0

with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,prediction=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(prediction==labels).sum().item()
    print(f"correctness in {(correct/total):.2f}")
