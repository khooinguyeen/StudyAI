import numpy as np
import matplotlib.pyplot as plt


numOfPoint = 30 # đặt số điểm dữ liệu là 30

noise = np.random.normal(0, 1, numOfPoint).reshape(-1, 1) 
# tạo nhiễu ngẫu nhiên bằng normal distribution với trung bình 0 và độ lệch chuẩn 1
# mảng được thay đổi kích thước để trở thành vector có kích thước (numOfPoint, 1)

x = np.linspace(30, 100, numOfPoint).reshape(-1, 1)
# tạo ra một mảng x chứa numOfPoint giá trị cách đều nhau từ 30 đến 100 (bao gồm cả hai đầu). 
# Hàm np.linspace được sử dụng để tạo ra các giá trị này.
# Mảng kết quả được thay đổi kích thước để trở thành một vector cột có kích thước (numOfPoint, 1).

N = x.shape[0] # gán số hàng trong mảng x cho biến N. Trong trường hợp này, N sẽ bằng numOfPoint.

y = 15*x + 8 + 20*noise 
# tính toán các giá trị y tương ứng dựa trên một phương trình tuyến tính. 
# Mảng y kết quả sẽ có cùng kích thước với x.

plt.scatter(x, y)
# tạo ra một biểu đồ phân tán bằng cách sử dụng hàm scatter của Matplotlib. 
# Nó nhận mảng x và y làm đầu vào và vẽ các điểm trên đồ thị. 
# Mỗi cặp (x, y) đại diện cho một điểm trên biểu đồ phân tán.

plt.show()

data = np.concatenate((x, y), axis=1)  # Ghép mảng x và y thành một mảng 2D

np.savetxt('data.csv', data, delimiter=',') 
# Lưu mảng data vào file data.csv với dấu phẩy (,) làm dấu phân cách 