# Linear Regression với sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('Linear Regression/data_linear.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

# Tạo mô hình hồi quy tuyến tính
lrg = LinearRegression()

# Train mô hình với data giá đất
lrg.fit(x, y)

# Đoán giá nhà đất 
y_pred = lrg.predict(x)

plt.plot((x[0], x[-1]), (y_pred[0], y_pred[-1]), 'r')
plt.show()

# Lưu nhiều tham số với numpy.savez(), định dạng '.npz'
np.savez('w2.npz', a=lrg.intercept_, b=lrg.coef_)

# Lấy lại các tham số trong file .npz
k = np.load('w2.npz')
lrg.intercept_ = k['a']
lrg.coef_ = k['b']
