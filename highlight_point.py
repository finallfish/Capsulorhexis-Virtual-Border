from skimage import io, measure
import cv2
import imutils
from imutils import contours
import numpy as np
import matplotlib.pyplot as plt


path = 'eyedata1/WLF/0122.jpg'
original_img = io.imread(path)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)       # 然后将其转换为灰度图并进行平滑滤波，以减少高频噪声
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]        # 阈值化处理，p>=200，设置为255(白色)
thresh = cv2.erode(thresh, None, iterations=2)              # 腐蚀
thresh = cv2.dilate(thresh, None, iterations=4)              # 膨胀

# labels = measure.label(thresh, neighbors=8, background=0)        # 连接组件分析 label存储的为阈值图像每一斑点对应的正整数
labels = measure.label(thresh, connectivity=2, background=0)        # 连接组件分析 label存储的为阈值图像每一斑点对应的正整数
mask = np.zeros(thresh.shape, dtype="uint8")                 # 初始化一个掩膜来存储大的斑点
plt.imshow(labels)
plt.show()

# 循环遍历每个label中的正整数标签，如果标签为零，则表示正在检测背景并可以安全的忽略它。否则，为当前区域构建一个掩码
for label in np.unique(labels):

    if label == 0:
        continue
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255

    orilabel = labelMask[100:300, 220:500]
    numori = cv2.countNonZero(orilabel)  # 初步选取一个块作为选区，选区外的不认
    plt.imshow(orilabel)
    plt.show()

    # 对labelMask中的非零像素进行计数。如果numPixels超过了一个预先定义的阈值 ,那么认为这个斑点“足够大”，并将其添加到掩膜中
    numPixels = cv2.countNonZero(labelMask)
    if 250 < numPixels < 600 and numori != 0:
        mask = cv2.add(mask, labelMask)

# plt.imshow(mask)
# plt.show()

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

(x, y, w, h) = cv2.boundingRect(cnts[0])
((cX, cY), radius) = cv2.minEnclosingCircle(cnts[0])
cv2.circle(original_img, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)

(point, radius) = cv2.minEnclosingCircle(cnts[0])
point = (int(point[0]), int(point[1]))
gray_img = gray_img[point[1] - 200:point[1] + 200, point[0] - 200:point[0] + 200]
# cv2.imwrite('result-image/point/DSQ0001.jpg',gray_img)

print((int(cX), int(cY)), radius)
plt.imshow(original_img, 'gray')
plt.title('src')
plt.show()


# 循环搜寻亮点的区间 并进行绘制
# for (i, c) in enumerate(cnts):
#
#     (x, y, w, h) = cv2.boundingRect(c)
#     ((cX, cY), radius) = cv2.minEnclosingCircle(c)
#     cv2.circle(original_img, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
#     cv2.putText(original_img, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


# sobelx = cv2.Sobel(gray_img,cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
# sobelXY = cv2.Sobel(gray_img, cv2.CV_64F, 1, 1, ksize=3)
# plt.subplot(2,2,1)
# plt.imshow(gray_img,'gray')
# plt.title('src')
# plt.subplot(2,2,2)
# plt.imshow(sobelx,'gray')
# plt.title('sobelX')
# plt.subplot(2,2,3)
# plt.imshow(sobely,'gray')
# plt.title('sobelY')
# plt.subplot(2,2,4)
# plt.imshow(sobelXY,'gray')
# plt.title('sobelXY')
# plt.show()



# 生成中间过程图
# path = 'eyedata1/DSQ/0190.jpg'
# original_img = io.imread(path)
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('result-image/point/1gray_img.jpg', gray_img)
#
# blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)       # 然后将其转换为灰度图并进行平滑滤波，以减少高频噪声
# cv2.imwrite('result-image/point/2lvbo.jpg', blurred)
#
# thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]        # 阈值化处理，p>=200，设置为255(白色)
# cv2.imwrite('result-image/point/3erzhihua.jpg', thresh)
#
# thresh = cv2.erode(thresh, None, iterations=2)              # 腐蚀
# cv2.imwrite('result-image/point/4fushi.jpg', thresh)
#
# thresh = cv2.dilate(thresh, None, iterations=4)              # 膨胀
# cv2.imwrite('result-image/point/5pengzhang.jpg', thresh)
#
#
# labels = measure.label(thresh, connectivity=2, background=0)        # 连接组件分析 label存储的为阈值图像每一斑点对应的正整数
# mask = np.zeros(thresh.shape, dtype="uint8")                 # 初始化一个掩膜来存储大的斑点
#
#
# # 循环遍历每个label中的正整数标签，如果标签为零，则表示正在检测背景并可以安全的忽略它。否则，为当前区域构建一个掩码
# for label in np.unique(labels):
#
#     if label == 0:
#         continue
#     labelMask = np.zeros(thresh.shape, dtype="uint8")
#     labelMask[labels == label] = 255
#     orilabel = labelMask[100:300, 220:500]
#     numori = cv2.countNonZero(orilabel)  # 初步选取一个块作为选区，选区外的不认
#     # 对labelMask中的非零像素进行计数。如果numPixels超过了一个预先定义的阈值 ,那么认为这个斑点“足够大”，并将其添加到掩膜中
#     numPixels = cv2.countNonZero(labelMask)
#     if 250 < numPixels < 600 and numori != 0 :
#         mask = cv2.add(mask, labelMask)
#
# cv2.imwrite('result-image/point/6mask.jpg', mask)
#
#
# # find the contours in the mask, then sort them from left to
# # right
# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = contours.sort_contours(cnts)[0]
#
# (x, y, w, h) = cv2.boundingRect(cnts[0])
# ((cX, cY), radius) = cv2.minEnclosingCircle(cnts[0])
# cv2.circle(original_img, (int(cX), int(cY)), int(radius), (255, 0, 0), 3)
#
#
# original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
# cv2.imwrite('result-image/point/7result.jpg', original_img)