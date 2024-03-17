import matplotlib.pyplot as plt
import numpy as np

# 坐标点
points = [(0, 0), (12500, -38), (14500, -9), (50000, -5)]

# 提取横坐标和纵坐标
x, y = zip(*points)

# 在中间插入震荡点
middle_x = (x[1] + x[2]) / 2
middle_y = np.mean([y[1], y[2]])  # 计算中间点的平均值

# 转换为列表，插入震荡点，再转换回元组
x = list(x)
y = list(y)
x.insert(2, middle_x)
y.insert(2, middle_y + 10)  # 调整震荡点的纵坐标，增大震荡幅度
x = tuple(x)
y = tuple(y)

# 重新排序坐标点
sorted_indices = np.argsort(x)
x = np.array(x)[sorted_indices]
y = np.array(y)[sorted_indices]

# 画曲线，设置颜色为蓝色
plt.plot(x[:2], y[:2], color='blue')  # 前两个点使用蓝色
plt.plot(x[1:3], y[1:3], color='red')  # 中间两个点使用红色
plt.plot(x[2:], y[2:], color='blue')  # 最后两个点使用蓝色

# 设置纵坐标范围和标题
plt.ylim(-40, 0)
plt.ylabel('Average Q-Value',fontsize=25)

# 设置横坐标范围和标题
plt.xlim(0, 50000)
plt.xlabel('Training Steps',fontsize=25)

plt.title('Training Step',fontsize=30)

# 添加图例
# plt.legend(fontsize=18)

# 添加格子状的背景颜色
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

# 显示图形
plt.show()
