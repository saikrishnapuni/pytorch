import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,3)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(3,1)
        # self.layer4 = nn.ReLU()
        # self.layer5 = nn.Linear(1,1)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        return x
    
    
distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5]
], dtype=torch.float32)

mean  = torch.mean(distances)
std = torch.std(distances)
transformed_distance = (distances-mean)/std

# print(transformed_distance)
times = torch.tensor([[3*i[0].item()+2 ] for i in distances], dtype=torch.float32)
# print(times)
# [92.98]
model = LinearRegression()
loss_function = nn.MSELoss()
otpi = optim.Adam(model.parameters(),lr = 0.1)
for epoch in range(0,500):
    model.zero_grad()
    outputs = model(transformed_distance)
    loss  =  loss_function(times,outputs)
    loss.backward()
    otpi.step()

print(loss.item())

with torch.no_grad():
    new_data = torch.tensor([[20.0]], dtype=torch.float32)
    new_data = (new_data-mean)/std
    pred = model(new_data)
print(f'original:{62} and predicted:{pred[0].item()}')
print(model.parameters())
        