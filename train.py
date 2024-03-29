import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model import Net
import argparse
from hyperparams import _base_path


parser = argparse.ArgumentParser(description="Process hyper-parameters")
parser.add_argument("--epochs", type=int, default=1, help="epochs of training")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root=f"{_base_path}/data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
test_dataset = torchvision.datasets.MNIST(
    root=f"{_base_path}/data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

# Create data loaders
train_loader: DataLoader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the network
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# model parameters
num_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {num_params}")

# Train the network for 1 epochs
for epoch in range(args.epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print statistics
        if (i + 1) % 10000 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, 3, i + 1, len(train_loader), loss.item()
                )
            )
    checkpoint = {
        "epoch": epoch,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{_base_path}/mnist_net_checkpoint.pth")

# Evaluate the network on the test set
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy: {}%".format((correct / total) * 100))

# Save the model
torch.save(net.state_dict(), f"{_base_path}/mnist_net.pth")
