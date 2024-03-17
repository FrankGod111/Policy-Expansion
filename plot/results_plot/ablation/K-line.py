import matplotlib.pyplot as plt

# 数据
scene_types = ["h-r", "h-m", "h-m-r",
               "ho-ra", "ho-m", "ho-m-ry",
               "w-r", "w-m", "w-m-r"]
# exploration6
percent_differences = [13, -22, -40, 8, 1.9, 18.4, 0.8, -4.9, 1.1]

# exploration 5
# percent_differences = [-40.32, -20.12, -40.01, -26.73, -40.15, -7.28, -13.07, -40.59, -32.44]
# exploration 4
# percent_differences = [8.3, 6.9, -28.1, 6.5, 2.1, -7.2, 2.4, -5.8, -8.2]
# exploration 3
# percent_differences = [-1.17, -24.67, -2.78, 4.33, -3.48, -7.18, -1.89, 2.12, 3.29]
# exploration 2
# percent_differences = [1.5, -13.5, 2.3, 0.7, 4.2, -3.7, 1.4, 2.1, -1.4]
# exploration 1
# percent_differences = [-2.4, 9.7, -7.1, -4.2, -5.3, -23.8, -1.2, -2.6, -2.1]

# 颜色映射：值大于0的使用浅绿色，值小于0的使用浅粉色
colors = ['lightgreen' if pd >= 0 else 'lightpink' for pd in percent_differences]

# 创建柱状图
plt.figure(figsize=(20, 15))
plt.barh(scene_types, percent_differences, color=colors)

# 添加标题和标签，以及调整字体大小
plt.title("Exploration", fontsize=50)
plt.xlabel("Percent Difference", fontsize=45)
# plt.ylabel("Scene Type", fontsize=40)

# 设置坐标轴刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=25)

# 设置刻度标签的字体大小
plt.xticks(fontsize=40)
plt.yticks(fontsize=45)

# 添加图例
#plt.legend(fontsize=18)

# 添加网格
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 设置淡线条的大网格

# 显示图表
plt.show()
