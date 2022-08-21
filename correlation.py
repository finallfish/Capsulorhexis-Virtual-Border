from skimage import io,color
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from time import *

# np.set_printoptions(threshold=10000000)

path = 'eyedata/DSQ/0001.jpg'
original_img=io.imread(path)
gray_img= cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)


# kernal = np.array([79,78,77,76,75,66,65,60,59,60,61])
kernal = np.array([71,68,70,60,62,49,53,52,56,56,57,58,61,62,62])
kernal_test = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])


begin_time = time()

w,h = gray_img.shape
edge_img = np.zeros([w,h])
for i in range(w-14):
    for j in range(h):
        for k in range(15):
            kernal_test[k] = gray_img[i+k][j]      # 从图像中取列的15个数
        if np.cov(kernal_test)==0:
            kernal_test[14] = kernal_test[14]+1          # 防止分母为0
        coefficient = np.corrcoef(kernal,kernal_test)        # 计算相关矩阵
        edge_img[i][j] = coefficient[0][1]                # 取出相关系数

        if coefficient[0][1]==1:
            print('kernal=', kernal)
            print('kernal_test=', kernal_test)
            print('coefficient[0][1]=',coefficient[0][1])

# edge_img = edge_img * 255
end_time = time()
run_time = end_time - begin_time
# print(edge_img)
print('程序运行的时间：', run_time)

plt.imshow(edge_img,cmap='gray')
plt.show()


edge_img = edge_img.astype(np.uint8)
circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=40, minRadius=80, maxRadius=120)

# 整数化，#把circles包含的圆心和半径的值变成整数
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # if(i[1]>50 and i[1]<125 and i[0]<125 and i[0]>50):
        cv2.circle(gray_img, (i[0], i[1]), i[2], 255, 1)
        cv2.circle(gray_img, (i[0], i[1]), 2, 255, 1)


print("圆心坐标", i[0], i[1])
plt.subplot(121)
plt.imshow(gray_img, cmap='gray')
plt.subplot(122)
plt.imshow(edge_img, cmap='gray')
plt.show()


