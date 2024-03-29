from copy import deepcopy
from torch.nn.utils import parameters_to_vector
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Net
from hyperparams import _n_params_subnet, _base_path


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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

net = Net().to(device)
net.load_state_dict(torch.load(f"{_base_path}/mnist_net.pth"))

n_snapshots = 0
n_snapshots_total = 2
snapshot_freq = 1
lr = 0.01
momentum = 0.9
weight_decay = 3e-4
min_var = 1e-30
n_epochs = snapshot_freq * n_snapshots_total

_model = deepcopy(net)
_model.train()


def _param_vector(model):
    return parameters_to_vector(model.parameters()).detach()


mean = torch.zeros_like(_param_vector(_model))
sq_mean = torch.zeros_like(_param_vector(_model))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    _model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)

N = len(train_dataset)
for epoch in range(n_epochs):
    i = 0
    for inputs, targets in train_loader:
        i = i + 1
        if i % N == 0:
            print(f"epoch: {epoch}; data: {i}/{N}")

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(_model(inputs), targets)
        loss.backward()
        optimizer.step()

    if epoch % snapshot_freq == 0:
        old_fac, new_fac = n_snapshots / (n_snapshots + 1), 1 / (n_snapshots + 1)
        mean = mean * old_fac + _param_vector(_model) * new_fac
        sq_mean = sq_mean * old_fac + _param_vector(_model) ** 2 * new_fac
        n_snapshots += 1

param_variances = torch.clamp(sq_mean - mean**2, min_var)
idx = torch.argsort(param_variances, descending=True)[:_n_params_subnet]
idx = idx.sort()[0]
parameter_vector = parameters_to_vector(net.parameters()).detach()
subnet_mask = torch.zeros_like(parameter_vector).bool()
subnet_mask[idx] = 1
subnet_mask_indices = subnet_mask.nonzero(as_tuple=True)[0]

torch.save(subnet_mask_indices, "subnet_mask_indices.pt")
