import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)


model = Net()
print('训练前需要梯度的参数：')
for name, param in model.named_parameters():
    print(name, param.requires_grad)

for param in model.fc1.parameters():
    param.requires_grad = False

print("\n冻结 fc1 后需要梯度的参数:")
for name, param in model.named_parameters():
    print(name, param.requires_grad)