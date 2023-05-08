import numpy as np
import matplotlib.pyplot as plt


numOfPoint = 30
noise = np.random.normal(0, 1, numOfPoint).reshape(-1, 1)
print(noise)
x = np.linspace(30, 100, numOfPoint).reshape(-1, 1)
print(x)
# N = x.shape[0]
# y = 15*x + 8 + 20*noise
# plt.scatter(x, y)

# plt.show()