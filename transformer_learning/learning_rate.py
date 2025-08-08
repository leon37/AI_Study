import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split

def f(x, y):
    return x**2+2*y**2

num_samples = 1000
X = torch.rand(num_samples)
Y = torch.rand(num_samples)

Z = f(X, Y)+3*torch.randn(num_samples)
dataset = torch.stack([X, Y, Z], dim=1)

train_size = int(0.7*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset=dataset, length=[train_size, test_size])

train_dataloader = DataLoader(TensorDataset(train_dataset.dataset.narrow(1, 0, 2), train_dataset.dataset.narrow(1, 2, 1)), batch_size=32)
test_dataloader = DataLoader(TensorDataset(test_dataset.dataset.narrow(1, 0, 2), test_dataset.dataset.narrow(1, 2, 1)), batch_size=32)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

num_epochs = 100
learning_rate = 0.1

loss_fn = nn.MSELoss()

for with_scheduler in (True, False):
    train_losses = []
    test_losses = []

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)
        if with_scheduler:
            scheduler.step()

    plt.figure(figsize=[8,4])
    plt.plot(range(num_epochs), train_losses, label='train')
    plt.plot(range(num_epochs), test_losses, label='test')
    plt.title("{0} lr_schedule".format("With" if with_scheduler else "Without"))
    plt.legend()

    plt.show()
