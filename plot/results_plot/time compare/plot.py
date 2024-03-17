import matplotlib.pyplot as plt

# 数据
categories = ['TD3+BC', 'AWAC', 'IQL', 'MAERL', 'CQL']
data = [1.76, 2.36, 2.13, 4.8, 6.7]  # 这里是初始的数据，你可以根据需要修改
colors = ['skyblue', 'steelblue', 'darkseagreen', 'indianred', 'darkorange']  # 每个柱状对应的颜色，你可以根据需要修改

# 创建横向柱状图
plt.figure(figsize=(8, 6))
plt.barh(categories, data, color=colors)

# 设置横纵坐标标题
plt.xlabel('Time (h)', fontsize=20)
plt.ylabel('')

# 设置图的标题
plt.title('Comparison of Experiment Time', fontsize=20)

# 添加格子状的背景颜色
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

# 设置坐标轴刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=18)

# 添加图例
plt.legend(fontsize=16)

# 显示图表
plt.show()
