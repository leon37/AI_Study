import numpy as np
import matplotlib.pyplot as plt

d = 512
n_data = 2000
np.random.seed(0)
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype(np.float32)

plt.hist(data[5])
plt.show()