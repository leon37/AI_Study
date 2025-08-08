import torch
import numpy as np
import torch.nn as nn

np.random.seed(42)

x = np.random.rand(100, 1)
y = 1 + 2*x + 0.1*np.random.randn(100, 1)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()
learning_rate = 0.1
num_epochs = 1000

input_dim = 1
output_dim = 1

model = nn.Linear(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w: ', model.weight.data)
print('b: ', model.bias.data)