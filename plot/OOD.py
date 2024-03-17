import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 生成模拟的高斯分布数据
mu, sigma = 0, 0.1
data1 = np.random.normal(mu, sigma, 1000)
data2 = np.random.normal(mu - 0.08, sigma, 1000)

# 创建柱状图
plt.figure(figsize=(8, 6))
n1, bins1, _ = plt.hist(data1, bins=30, color='gray', edgecolor='lightgray', alpha=0.7, label='Offline Distribution')
n2, bins2, _ = plt.hist(data2, bins=30, color='lightblue', edgecolor='lightgray', alpha=0.7, label='Online Distribution')

# 找到柱状图的最高点
x1 = (bins1[:-1] + bins1[1:]) / 2
x2 = (bins2[:-1] + bins2[1:]) / 2
max_index1 = np.argmax(n1)
max_index2 = np.argmax(n2)

# 连接最高点并进行曲线拟合
x_points = [x1[max_index1], x2[max_index2]]
y_points = [n1[max_index1], n2[max_index2]]
z = np.polyfit(x_points, y_points, 1)
p = np.poly1d(z)



# 设置图的标题
plt.title('Halfcheetah',fontsize=20)

# 设置横坐标和纵坐标标题
plt.xlabel('Log-likelihood',fontsize = 20)
plt.ylabel('Normalization Frequency',fontsize = 20)

# 添加图例
plt.legend(fontsize=15)

# 设置背景颜色为白色
plt.gca().set_facecolor('white')

# 显示图表
plt.show()
