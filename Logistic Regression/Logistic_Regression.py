# Logistic Regression dùng thư viện sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Load data từ file csv
data = pd.read_csv('Logistic Regression/dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
# Vẽ data bằng scatter
plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')
# Tạo mô hình Logistic Regression và train
logreg = LogisticRegression()
logreg.fit(x, y)
# Lưu các biến của mô hình vào mảng
wg = np.zeros( (3, 1) )
wg[0, 0] = logreg.intercept_
wg[1:, 0] = logreg.coef_
# Vẽ đường phân cách
t = 0.5
plt.plot((4, 10),(-(wg[0]+4*[1]+ np.log(1/t-1))/wg[2], \
-(wg[0] + 10*wg[1]+ np.log(1/t-1))/wg[2]), 'g')
plt.show()