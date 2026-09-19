import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader,random_split,Subset
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# mean_std_transform = transforms.Compose([
#     transforms.Resize((256,256)) ,
#     transforms.CenterCrop(224) ,
#     transforms.ToTensor()
# ])

dataset = datasets.ImageFolder(
    "seg_train/seg_train",
    # transform=mean_std_transform
)

# loader = DataLoader(dataset,batch_size=64,shuffle=False)
# mean = torch.zeros(3)
# std = torch.zeros(3)
# total_images =0

# for images,_ in loader:
    
#     batch_size = images.size(0)

#     images = images.view(batch_size, 3, -1)

#     mean += images.mean(dim=2).sum(dim=0)
#     std += images.std(dim=2).sum(dim=0)

#     total_images += batch_size
mean = [0.4346, 0.4610, 0.4557]
std = [0.2207, 0.2191, 0.2261]
print(mean,std)

 

training_transform = transforms.Compose([
    transforms.Resize((256,256)) ,
    transforms.CenterCrop(224) ,
    transforms.RandomHorizontalFlip(0.5),
    
    
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
    
])

validation_transform = transforms.Compose([
    transforms.Resize((256,256)) ,
    transforms.CenterCrop(224) ,
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])

train_full = datasets.ImageFolder(
    "seg_train/seg_train",
    transform=training_transform
)

val_full = datasets.ImageFolder(
    "seg_train/seg_train",
    transform=validation_transform
)       

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size

train_indices, val_indices = random_split(
    range(len(train_full)),
    [train_size, val_size]
)

train = Subset(train_full, train_indices.indices)
validation = Subset(val_full, val_indices.indices)


def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Model().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

train = DataLoader(train, batch_size=64, shuffle=True)
validation = DataLoader(validation, batch_size=64, shuffle=False)


epochs = 10

for i in range(epochs):
    model.train()
    rl = 0
    c = 0
    t = 0
    for data,target in train:
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()
        rl = rl+loss.item()
        t += target.size(0)
        _,pred = output.max(1)
        c+=pred.eq(target).sum().item()
    rl /= len(train)
    acc = 100.*(c/t)
    model.eval()
    vl = 0
    vc = 0
    vt = 0
    with torch.no_grad():
        for data,target in validation:
            data,target = data.to(device),target.to(device)
            output = model(data)
            loss = loss_function(output,target)
            vl = vl+loss.item()
            vt+=target.size(0)
            _,pred = output.max(1)
            vc+=pred.eq(target).sum().item()
        vl /= len(validation)
        vacc = 100.*(vc/vt)
    print(
        f"Epoch {i + 1}/{epochs} | "
        f"Train Loss: {rl:.4f} | "
        f"Train Acc: {acc:.2f}% | "
        f"Val Loss: {vl:.4f} | "
        f"Val Acc: {vacc:.2f}%"
    )
        