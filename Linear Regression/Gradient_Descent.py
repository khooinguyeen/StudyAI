import numpy as np
import matplotlib.pyplot as plt

# TÌM MIN CỦA Y=X^2 BẰNG GRADIENT DESCENT

x = 10
y = []

for i in range(10):
    x = x - 0.1 * 2 * x # 8.0 6.4 5.12 4.096 3.2768 2.62144 2.0971520000000003 1.6777216000000004 1.3421772800000003 1.0737418240000003
    y.append(x**2) # mũ 2 
    # [64.0, 40.96000000000001, 26.2144, 16.777216, 10.73741824, 6.871947673600001, 4.398046511104002, 2.8147497671065613, 1.801439850948199, 1.1529215046068475]

plt.plot(y)
plt.xlabel('Số lần')
plt.ylabel('f(x)')
plt.title('Giá trị f(x) sau số lần thực hiện bước 2')
plt.show()