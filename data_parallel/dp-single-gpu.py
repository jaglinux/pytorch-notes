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
        return torch.randn(10), torch.randn(1)

    def __len__(self):
        return self.length


# Instantiate model, dataset, and dataloader
model = SimpleModel()
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    #model = nn.DataParallel(model)
dataset = DummyDataset(1000)
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
