import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# path = 'eyedata/DSQ/yuan.png'
path = 'eyedata/DSQ/0001.jpg'
# excel = 'eyedata/DSQ/1.xlsx'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray,cmap="gray")
# plt.show()

# data = pd.DataFrame(gray)
# writer = pd.ExcelWriter(excel)		# 写入Excel文件
# data.to_excel(writer, 'DSQ', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()


# # Sobel边缘检测算子
# img = cv2.imread(path, 0)
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
# Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
# Scale_absY = cv2.convertScaleAbs(y)
# result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
# cv2.imshow('img', img)
# cv2.imshow('Scale_absX', Scale_absX)
# cv2.imshow('Scale_absY', Scale_absY)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Scharr算子
# img = cv2.imread(path, 0)
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=-1)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=-1)
# # ksize=-1 Scharr算子
# # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
# Scharr_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
# Scharr_absY = cv2.convertScaleAbs(y)
# result = cv2.addWeighted(Scharr_absX, 0.5, Scharr_absY, 0.5, 0)
# cv2.imshow('img', img)
# cv2.imshow('Scharr_absX', Scharr_absX)
# cv2.imshow('Scharr_absY', Scharr_absY)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 拉普拉斯算子
# img = cv2.imread(path, 0)
# blur = cv2.GaussianBlur(img, (3, 3), 0)
# laplacian = cv2.Laplacian(blur, cv2.CV_16S, ksize=5)
# dst = cv2.convertScaleAbs(laplacian)
# result = dst
# cv2.imshow('laplacian', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# canny算子
img = cv2.imread(path, 0)
# img = cv2.resize(img,(200,200))
blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪

canny = cv2.Canny(blur, 0, 20)  # 50是最小阈值,150是最大阈值
result = canny
# cv2.imshow('canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# circles1 = cv2.HoughCircles(result,cv2.HOUGH_GRADIENT,1, 600,param1=50,param2=30,minRadius=0,maxRadius=0)
# circles = circles1[0,:,:]
# circles = np.uint16(np.around(circles))
# for i in circles[:]:
#     cv2.circle(img, (i[0], i[1]), i[2], 255, 1)

# for circles in circles1[:]:
#     circles = np.uint16(np.around(circles))
#     for i in circles[:]:
#         cv2.circle(img, (i[0], i[1]), i[2], 255, 1)

circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1, 600, param1=50, param2=40, minRadius=0, maxRadius=0)
# 整数化，#把circles包含的圆心和半径的值变成整数
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    if(i[1]>50 and i[1]<125 and i[0]<125 and i[0]>50):
        cv2.circle(img, (i[0], i[1]), i[2], 255, 1)
        cv2.circle(img, (i[0], i[1]), 2, 255, 1)


print("圆心坐标", i[0], i[1])
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(result)
plt.show()
