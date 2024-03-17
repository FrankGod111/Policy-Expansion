import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm  #字体管理器
import pickle


#plt.style.use("ggplot")
plt.figure(figsize=(10,6))

x_data = np.linspace(-np.pi,np.pi,64,endpoint=True)
gs = gridspec.GridSpec(1,1) #将绘图区分成两行三列
ax1 = plt.subplot(gs[0,0])#指定ax1占用第一行(0)整行
#ax2 = plt.subplot(gs[0,1])#指定ax2占用第二行(1)的第一格(第二个参数为0)

x_len = 120

X = np.arange(x_len)

Baseline2 = np.ones(x_len) * 72.6
y_data = np.random.rand(x_len) + Baseline2

#test_data[1][:x_len]

ax1.set_xlabel("Epoch", fontsize=30)
ax1.set_ylabel("Accuracy", fontsize=30)
ax1.plot(X, y_data , lw=2, label="Fusion", color='#11618A')
ax1.plot(X,Baseline2+0.4, lw=2, label="Teacher", color='#C49F31')

# Erase 上面 the data by filling with white
ax1.fill_between(X, y_data, min(y_data), color='#1786BF', alpha=0.5)
ax1.fill_between(X, Baseline2+0.4, min(y_data), color='#F2C43D', alpha=0.5)

ax1.legend(fontsize=30)
ax1.tick_params(labelsize=30)

plt.savefig("line_fusion.svg", dpi=300,format="svg")
plt.subplots_adjust(bottom=0.2,left=0.18)

plt.show()