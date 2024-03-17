import matplotlib.pyplot as plt
import random

# 生成随机数据点
x = [i for i in range(51)]  # 横坐标范围（0, 50）
y = [random.randint(4500, 5100) for _ in range(51)]  # 纵坐标范围（4500, 5100）

# 创建折线图
plt.figure(figsize=(8, 6))
plt.plot(x, y, color='blue', linewidth=2, label='Random Line')

# 设置横坐标和纵坐标范围
plt.xlim(0, 50)
plt.ylim(3500, 9000)  # 调整纵坐标范围

# 设置 y 轴刻度
plt.yticks(range(2000, 10000, 1000))

# 添加水平线
plt.axhline(y=6800, color='brown', linestyle='--', linewidth=2, label='y=6800')  # 棕色虚线
plt.axhline(y=8600, color='deeppink', linestyle='--', linewidth=2, label='y=8600')  # 深粉红虚线

# 生成新的随机数据点
y_new = [random.randint(3500, 6000) for _ in range(51)]  # 使红色线条整体缓慢递增且带有曲折的折线

# 添加新的折线图
plt.plot(x, y_new, color='red', linewidth=2, label='Random Line (Red)')

# 设置图的标题
plt.title('Random Line Chart')

# 设置横坐标和纵坐标标题
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加图例
plt.legend()

# 设置网格样式
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 设置淡线条的大网格

# 显示图表
plt.show()
