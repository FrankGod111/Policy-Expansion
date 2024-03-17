import matplotlib.pyplot as plt

# 横坐标取值
x = [0.0, 0.2, 0.4, 0.8, 1.0]

# 纵坐标取值
q_first = [57.29, 59.76, 67.68, 73.76, 75.98]
q1 = [57.29, 58.5, 59.3, 60.1, 61.0]
q2 = [57.29, 58.2, 59.5, 60.8, 61.7]
q3 = [57.29, 59.0, 62.1, 65.3, 67.8]
q4 = [57.29, 58.8, 61.5, 64.2, 66.9]
q5 = [57.29, 59.2, 62.7, 66.4, 69.1]

# 设置不同的线条颜色
plt.plot(x, q4, label=r'$\alpha=0.1$', marker='o', color='purple')

plt.plot(x, q3, label=r'$\alpha=0.5$', marker='o', color='blue')
plt.plot(x, q2, label=r'$\alpha=0.8$', marker='o', color='green')
plt.plot(x, q_first, label=r'$\alpha=1$', marker='o', color='orange')
plt.plot(x, q1, label=r'$\alpha=2$', marker='o', color='red')
plt.plot(x, q5, label=r'$\alpha=5$', marker='o', color='brown')

# 设置横坐标和纵坐标标题
plt.xlabel('Training Steps(M)', fontsize=20)
plt.ylabel('Normalized Return', fontsize=20)

# 设置坐标轴刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=18)

# plt.title('MEARL', fontsize=20)

# 添加图例
plt.legend(fontsize=15)

# 设置网格样式
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 设置淡线条的大网格
# 显示图表
plt.show()
