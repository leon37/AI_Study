import torch
from matplotlib import pyplot as plt


def f(x):
    return x**2+4*x+1

x = torch.tensor(-10.0, requires_grad=True)
learning_rate = 0.9

xs = []
ys = []

for i in range(100):
    y = f(x)
    xs.append(x.item())
    ys.append(y.item())

    y.backward()
    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()

print(f'最终参数值: {x.item()}')

x_origin = torch.arange(-10, 10, 0.1)
y_origin = f(x_origin)

plt.plot(x_origin, y_origin, 'b-')
plt.plot(xs, ys, 'r--')
plt.scatter(xs, ys, s=50, c='r')

plt.xlabel('x')
plt.ylabel('y')

plt.show()