from matplotlib import pyplot as plt
import random
import numpy as np
import copy
import cv2
import matplotlib


img = cv2.imread("eyedata1/DSQ/0001.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# line_row = gray[210][:]
line_row = gray[170][:]
line_column = np.zeros(576)
for t in range(576):
    line_column[t] = gray[t][400]

for i in range(720):
    img[169][i] = (0, 0, 255)
    img[170][i] = (0,0,255)
    img[171][i] = (0, 0, 255)
for i in range(576):
    img[i][399] = (255,0,0)
    img[i][400] = (255,0,0)
    img[i][401] = (255,0,0)
plt.imshow(img, cmap='gray')
# plt.savefig('1111.png')
plt.show()

x_row = range(0, 720)
plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x_row, line_row, color='b')
# plt.savefig('2222.png')
plt.show()
x_column = range(0, 576)
plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x_column, line_column, color='r')
# plt.savefig('3333.png')
plt.show()

# x = range(0, 120)
# y = [random.randint(20, 35) for i in range(120)]
# plt.figure(figsize=(20, 8), dpi=80)
# plt.plot(x, y)
# # 调整X轴的刻度
# # _x =list(x)[::10]   # X轴过于密集需要取步长 进行强制转换 转换成列表进行步长设置
# _xtick_labels = ["10点{}分".format(i) for i in range(60)]  # _x与_xtick_labels的步长数量一致
# _xtick_labels += ["11点{}分".format(i) for i in range(60)]
# # 把数值型数据对应到字符串类型 只有列表能取步长 要强制转换列表
# plt.xticks(list(x)[::3], _xtick_labels[::3], rotation=45)  # 追求一一对应 rotation旋转的度数 fontproperties是设置对应的字体参数
#
# plt.xlabel('时间')
# plt.ylabel('温度 单位(℃)')
# plt.title('10点到12点每分钟的气温变化情况')
# plt.show()