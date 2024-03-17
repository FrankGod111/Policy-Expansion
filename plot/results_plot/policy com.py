import matplotlib.pyplot as plt
import numpy as np

# 创建更多数据点
training_iterations = np.linspace(0, 1, num=1001) * 1e6

# 创建不对称的两条折线，使曲线更加稀疏和振幅逐渐减小
policy_usage1 = 0.23 + 0.15 * np.sin(0.05 * training_iterations) * np.exp(-0.000002 * training_iterations) + 0.1 * np.random.randn(1001)
policy_usage2 = 0.76 - 0.15 * np.sin(0.05 * training_iterations) * np.exp(-0.000002 * training_iterations) + 0.1 * np.random.randn(1001)

# 选择更少的要添加的点的位置
points_x = [0.2e6, 0.4e6, 0.6e6, 0.8e6, 1.0e6]
points_y1 = [0.45, 0.55, 0.65, 0.6, 0.46]
points_y2 = [0.65, 0.75, 0.65, 0.6, 0.53]

# 绘制图表
plt.figure(figsize=(10, 6))

plt.plot(training_iterations, policy_usage1, label='Line 1')
plt.plot(training_iterations, policy_usage2, label='Line 2')

# 添加更少的点并将其连接到曲线上
plt.scatter(points_x, points_y1, color='red', label='Points 1')
plt.scatter(points_x, points_y2, color='blue', label='Points 2')

for i in range(len(points_x)):
    plt.plot([points_x[i], points_x[i]], [points_y1[i], policy_usage1[i]], 'r--', linewidth=1)
    plt.plot([points_x[i], points_x[i]], [points_y2[i], policy_usage2[i]], 'b--', linewidth=1)

# 去掉背景线条
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置横坐标和纵坐标刻度
plt.xticks(np.arange(0, 1.1e6, 0.1e6), ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
plt.yticks(np.arange(0.2, 0.9, 0.1))

# 添加标签和标题
plt.xlabel('Training Iterations (1e6)')
plt.ylabel('Policy Usage')
plt.title('Asymmetric and Sparse Policy Usage Over Training Iterations')

# 添加图例
plt.legend()

# 显示图表
plt.show()
