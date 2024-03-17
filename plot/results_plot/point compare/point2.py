import matplotlib.pyplot as plt
import numpy as np

# 生成随机散点数据
np.random.seed(42)
num_points = 100
x_values = np.linspace(-2, 4, num_points)  # 在-2到4之间生成等间隔的值
y_values = np.random.uniform(0, 4, num_points)
colors = y_values / 4.0  # 根据y值大小来设置颜色，越大颜色越深

# 绘制散点图
plt.scatter(x_values, y_values, c=colors, cmap='Blues', s=50, alpha=0.7)

# 设置横纵坐标标题
plt.xlabel('Gap assessment', fontsize=20)
plt.ylabel('')

# 设置图的标题
plt.title('(b) 50 epochs', fontsize=24)

# 添加格子状的背景颜色
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

# 在横坐标为1、4处添加虚线
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=4, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=7, color='gray', linestyle='--', linewidth=1)
# 显示图表
plt.show()
