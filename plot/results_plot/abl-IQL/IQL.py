import matplotlib.pyplot as plt

# 横坐标取值
x = [10, 20, 40, 80]

# 纵坐标取值
uncertainty_first = [78.16, 75.29, 76.61, 77.75]
q_first = [76.52, 75.34, 75.59, 76.92]
best_baseline = [73, 73, 73, 73]

# 绘制三条线
plt.plot(x, uncertainty_first, label='Uncertainty first', marker='o')
plt.plot(x, q_first, label='Q first', marker='o')
plt.plot(x, best_baseline, label='Best Baseline', linestyle='--', marker='o')  # 使用虚线样式

# 设置横坐标和纵坐标标题
plt.xlabel('K',fontsize=20)
plt.ylabel('Average D4RL score',fontsize=20)

# 设置坐标轴刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=18)

plt.title('MAERL(IQL)',fontsize=24)

# 添加图例
plt.legend(fontsize=16,loc = 'lower right')

# 设置网格样式
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 设置淡线条的大网格
# 显示图表
plt.show()
