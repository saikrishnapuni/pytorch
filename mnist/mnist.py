import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',train=True,download=True,transform=transform)

test_dataset = torchvision.datasets.MNIST(
    root='./data',train=False,download=True,transform=transform)
# print(train_dataset[0][0].shape)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)

class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten1 = nn.Flatten()
        self.Layers = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.flatten1(x)
        x = self.Layers(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Mnist().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)
def train_epoch(model,train_loader,loss_function,optimizer,device):
    model.train()
    rl = 0.0
    c = 0
    t = 0
    for b_id,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()
        rl += loss.item()
        t += target.size(0)
        _,predicted = output.max(1)
        c+=predicted.eq(target).sum().item()
        if(b_id%100 == 0 and b_id>0):
            avg = rl/100
            acc = 100.*c/t
            print(b_id,acc,avg)
            rl = 0
def evaluate(model,test_loader,device):
    model.eval()
    c = 0
    tot = 0
    with torch.no_grad():
        for i,t in test_loader:
            i,t = i.to(device),t.to(device)
            o = model(i)
            _,p = o.max(1)
            c+=p.eq(t).sum().item()
            tot += t.size(0)
    return 100.*c/tot
for epoch in range(10):
    print("epoch no ,",epoch+1)
    train_epoch(model,train_loader,loss_function,optimizer,device)
    acc = evaluate(model,test_loader,device)
    print("Test Accuracy : ",acc)
