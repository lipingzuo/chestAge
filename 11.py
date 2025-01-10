import numpy as np
import matplotlib.pyplot as plt

# 定义Leaky ReLU函数
def leaky_relu(x, alpha=0.02):
    return np.where(x > 0, x, alpha * x)

# 生成输入数据
x = np.linspace(-10, 10, 400)
y = leaky_relu(x)

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Leaky ReLU Function', color='blue')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Leaky ReLU Activation Function')
plt.grid(True)
plt.legend()
plt.show()
