# this code is without DP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create a dummy dataset
class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        # This will be multiplied by batch_size
        return torch.randn(10), torch.randn(1)

    def __len__(self):
        # total number of datasets
        return self.length


# Instantiate model, dataset, and dataloader
model = SimpleModel()
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    #model = nn.DataParallel(model)
dataset = DummyDataset(1000)
# for every iteration, batch_size * torch.randn(10), batch_size * torch_randn(1) will be returned
# 64 * 10 , 64 * 1
# total number of iterations in an epoch is len(dataset) / batch_size
# 1000 / 64 = 16. Last iteration will have less batch size
dataloader = DataLoader(dataset, batch_size=64)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device, torch.cuda.get_device_name(device))
model.to(device)

device_type = next(model.parameters()).device
print("model is on ", device_type)

# Training loop
for epoch in range(10):
    iteration = 0
    for x, y in dataloader:
        print("epoch is ", epoch, " iteration is ", iteration)
        iteration+=1
        x, y = x.to(device), y.to(device)
        print("x and y shapes are ",x.shape, y.shape)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

#o/p (using only 2 epochs)
'''
cuda AMD Instinct MI250X/MI250
model is on  cuda:0
epoch is  0  iteration is  0
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  1
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  2
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  3
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  4
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  5
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  6
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  7
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  8
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  9
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  10
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  11
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  12
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  13
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  14
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  0  iteration is  15
x and y shapes are  torch.Size([40, 10]) torch.Size([40, 1])
Epoch 1, Loss: 1.5829228162765503
epoch is  1  iteration is  0
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  1
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  2
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  3
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  4
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  5
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  6
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  7
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  8
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  9
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  10
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  11
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  12
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  13
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  14
x and y shapes are  torch.Size([64, 10]) torch.Size([64, 1])
epoch is  1  iteration is  15
x and y shapes are  torch.Size([40, 10]) torch.Size([40, 1])
Epoch 2, Loss: 1.2430177927017212

'''
