import numpy as np
import matplotlib.pyplot as plt

num = 30

noise = np.random.normal(0, 1, num).reshape(-1, 1)
# np.save('Linear Regression/noise.npy', noise)
# noise = np.load('Linear Regression/noise.npy')
# noise.reshape(-1, 1)
print(noise)