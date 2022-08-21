import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.filters import gaussian
from skimage.segmentation import active_contour

path = 'eyedata/DSQ/0001.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 圆的参数方程：(220, 100) r=100
t = np.linspace(0, 2*np.pi, 400) # 参数t, [0,2π]
x = 190 + 150*np.cos(t)
y = 200 + 150*np.sin(t)

# 构造初始Snake
init = np.array([x, y]).T # shape=(400, 2)

# Snake模型迭代输出
snake = active_contour(gaussian(img,3), snake=init, alpha=0.1, beta=1, gamma=0.01, w_line=0, w_edge=10)

# 绘图显示
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.plot(init[:, 0], init[:, 1], '--r', lw=3)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.xticks([]), plt.yticks([]), plt.axis("off")
plt.show()
