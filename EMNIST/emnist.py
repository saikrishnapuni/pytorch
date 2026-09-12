import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Dataset
train_dataset = torchvision.datasets.EMNIST(
    root='/data',
    train=True,
    download=True,
    transform=transform,
    split='letters'
)

test_dataset = torchvision.datasets.EMNIST(
    root='/data',
    train=False,
    download=True,
    transform=transform,
    split='letters'
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


# Model
class EMNIST(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten_layer = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 26)
        )

    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.layers(x)
        return x


# Device
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(device)


# Model
model = EMNIST().to(device)


# Loss and optimizer
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training function
def training(model, train_loader, loss_function, optimizer, device):

    model.train()

    c = 0
    t = 0

    for b_id, (data, target) in enumerate(train_loader):

        # Move to GPU
        data = data.to(device)
        target = target.to(device)

        # EMNIST Letters labels are 1-26
        # CrossEntropyLoss expects 0-25
        target = target - 1

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Loss
        loss = loss_function(output, target)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Prediction
        predicted = output.max(1)[1]

        # Accuracy
        t += target.size(0)

        c += predicted.eq(target).sum().item()

        if b_id % 100 == 0 and b_id > 0:
            print(
                "accuracy = ",
                100 * (c / t)
            )


# Evaluation function
def evaluate(model, test_loader, device):

    model.eval()

    c = 0
    t = 0

    with torch.no_grad():

        for b_id, (data, target) in enumerate(test_loader):

            data = data.to(device)
            target = target.to(device)

            # EMNIST Letters labels are 1-26
            target = target - 1

            # Forward pass
            output = model(data)

            # Prediction
            predicted = output.max(1)[1]

            t += target.size(0)

            c += predicted.eq(target).sum().item()

    return 100. * c / t


# Training
for epoch in range(10):

    print("Epoch no:", epoch + 1)

    training(
        model,
        train_loader,
        loss_function,
        optimizer,
        device
    )

    acc = evaluate(
        model,
        test_loader,
        device
    )

    print("Test Accuracy:", acc)