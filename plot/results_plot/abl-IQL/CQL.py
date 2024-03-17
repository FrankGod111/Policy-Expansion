import matplotlib.pyplot as plt

# 横坐标取值
x = [10, 20, 40, 80]

# 纵坐标取值
uncertainty_first = [72.09, 76.58, 74.16, 71.43]
q_first = [57.29, 59.76, 67.68, 73.76]
best_baseline = [68, 68, 68, 68]

# 绘制三条线
plt.plot(x, uncertainty_first, label='Uncertainty first', marker='o')
plt.plot(x, q_first, label='Q first', marker='o')
plt.plot(x, best_baseline, label='Best Baseline', linestyle='--', marker='o')  # 使用虚线样式

# 设置横坐标和纵坐标标题
plt.xlabel('K',fontsize=20)
plt.ylabel('Average D4RL score',fontsize=20)

# 设置坐标轴刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=18)

plt.title('MAERL(CQL)',fontsize=24)

# 添加图例
plt.legend(fontsize=18)

# 设置网格样式
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 设置淡线条的大网格
# 显示图表
plt.show()
