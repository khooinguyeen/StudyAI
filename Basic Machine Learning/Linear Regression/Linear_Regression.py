import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# tạo dữ liệu
# numOfPoint = 30
# noise = np.random.normal(0, 1, numOfPoint).reshape(-1, 1)
# x = np.linspace(30, 100, numOfPoint).reshape(-1, 1)
# N = x.shape[0]
# y = 15*x + 8 + 20*noise
# plt.scatter(x, y)

data = pd.read_csv('Linear Regression/data_linear.csv').values 
# Đọc nội dung của tệp CSV có tên là "data_linear.csv" bằng hàm read_csv của pandas. 
# Thuộc tính values được sử dụng để trích xuất dữ liệu dưới dạng một mảng NumPy và gán vào biến data.

N = data.shape[0] # số hàng trong mảng data, đại diện cho số điểm dữ liệu (30)
x = data[:, 0].reshape(-1, 1)
# Trích xuất các giá trị từ cột đầu tiên của mảng data và 
# điều chỉnh kích thước thành một vector cột bằng reshape. 
# Vector cột được gán vào biến x.

y = data[:, 1].reshape(-1, 1) # như code trên nma là cột thứ 2

plt.scatter(x, y) # tạo biểu đồ phân tán

plt.xlabel('mét vuông') # đặt nhãn cho trục x
plt.ylabel('giá') # đặt nhãn cho trục y

x = np.hstack((np.ones((N, 1)), x)) 
# Thêm một cột chứa các giá trị 1 vào mảng x bằng cách sử dụng hàm np.hstack, 
# mảng x được sửa đổi. 
# Thao tác này thường được thực hiện để tính đến thuật ngữ bias trong các mô hình hồi quy tuyến tính.


w = np.array([0.,1.]).reshape(-1, 1)
# Tạo một vector trọng số ban đầu w dưới dạng một vector cột, được khởi tạo với các giá trị [0, 1].

# => Tóm tắt, đoạn code đọc dữ liệu từ tệp CSV, tạo một biểu đồ phân tán các điểm dữ liệu, 
# thêm một cột giá trị 1 vào mảng đầu vào và khởi tạo một vector trọng số cho mô hình hồi quy tuyến tính

numOfIteration = 100 # số lần lặp
cost = np.zeros((numOfIteration, 1)) 
# tạo mảng cost với kích thước (100, 1) gồm toàn số 0, sử dụng để lưu giá trị của cost function khi train

learning_rate = 0.000001 # set learning rate, dùng để điều chỉnh tốc độ cập nhật của weights

# Gradient descent
for i in range(1, numOfIteration): # vòng lặp huấn luyện
    r = np.dot(x, w) - y 
    # Biến r được tính toán bằng hiệu của tích vô hướng giữa x và w và y. Đây là sai số dự đoán.

    cost[i] = 0.5*np.sum(r*r)
    # Gán giá trị của hàm chi phí (cost function) cho phần tử thứ i trong mảng cost. 
    # Hàm chi phí được tính bằng nửa tổng bình phương của r, đại diện cho bình phương sai số.

    w[0] -= learning_rate*np.sum(r) 
    # Cập nhật trọng số đầu tiên (w[0]) bằng cách trừ đi learning_rate rồi nhân với tổng của r.
    # correct the shape dimension
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:, 1].reshape(-1, 1))) 
    # Cập nhật trọng số thứ hai (w[1]) bằng cách trừ learning_rate nhân với tổng của tích element-wise giữa r và cột thứ hai của x.


predict = np.dot(x, w) 
# Dự đoán đầu ra predict bằng tích vô hướng giữa x và w. 
# Đây là dự đoán giá trị đầu ra dựa trên mô hình hồi quy tuyến tính đã huấn luyện.

plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
# Vẽ một đường thẳng trên biểu đồ, nối các điểm (x[0][1], predict[0]) và (x[N-1][1], predict[N-1]), 
# được biểu thị bằng màu đỏ ( 'r').

plt.show()

x1 = 50 
y1 = w[0] + w[1] * x1 # Tính toán đầu ra dự đoán y1 cho x1 bằng cách sử dụng trọng số được huấn luyện w.
print('Giá nhà cho 50m^2 là: ', y1)

# Lưu w với numpy.save(), định dạng '.npy'
np.save('Linear Regression/weight.npy', w)

# Đọc file '.npy' chứa tham số weight
w = np.load('Linear Regression/weight.npy')

# Linear Regression với sklearn
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Linear Regression/data_linear.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
# tải dữ liệu và chia thành biến đầu vào x và biến mục tiêu y.

plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

# Tạo mô hình hồi quy tuyến tính
lrg = LinearRegression() # tạo instance của class LinearRegression

# Train mô hình với data giá đất
lrg.fit(x, y)

# Đoán giá nhà đất 
y_pred = lrg.predict(x)
# method predict() dùng để tạo ra các dự đoán (y_pred) cho biến (x) bằng cách sử dụng mô hình hồi quy tuyến tính(lrg).

plt.plot((x[0], x[-1]), (y_pred[0], y_pred[-1]), 'r')
plt.show()
# vẽ các điểm dữ liệu gốc dưới dạng điểm scatter và giá trị dự đoán (y_pred) dưới dạng một đường màu đỏ. 
# Biểu đồ kết quả sẽ hiển thị đường hồi quy tuyến tính biểu thị mối quan hệ giữa biến đầu vào (x) 
# và giá trị mục tiêu dự đoán (y_pred).

# Lưu nhiều tham số với numpy.savez(), định dạng '.npz'
np.savez('w2.npz', a=lrg.intercept_, b=lrg.coef_)
# The intercept value (giá trị chặn) is saved as a, and the coefficient value (giá trị hệ số) is saved as b.

# Lấy lại các tham số trong file .npz
k = np.load('w2.npz')
lrg.intercept_ = k['a']
lrg.coef_ = k['b']

# Giá trị chặn (intercept) được gán cho lrg.intercept_, và giá trị hệ số (coefficient) được gán cho lrg.coef_. 
# Điều này cho phép bạn khôi phục lại các tham số đã được lưu trữ trước đó cho mô hình hồi quy tuyến tính.