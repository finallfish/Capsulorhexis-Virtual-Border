import os
import cv2
import codecs
import imutils
import numpy as np
from skimage import measure
from imutils import contours
import matplotlib.pyplot as plt
path = 'eyedata1/WLF/'
cX = 350
cY = 231

def getlabel(path):
    f = open(path, "r")
    labellist = []
    i = 0
    for line in f.readlines():
        i += 1
        a = line.replace("\n", "")
        b = a.split(",")
        t = (b[1], b[2])
        labellist.append(t)
    labellist = np.array(labellist)
    return labellist


def highlight_point(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)  # 然后将其转换为灰度图并进行平滑滤波，以减少高频噪声
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]  # 阈值化处理，p>=200，设置为255(白色)
    # thresh = cv2.erode(thresh, None, iterations=2)  # 腐蚀
    # thresh = cv2.dilate(thresh, None, iterations=4)  # 膨胀
    labels = measure.label(thresh, connectivity=2, background=0)  # 连接组件分析 label存储的为阈值图像每一斑点对应的正整数
    mask = np.zeros(thresh.shape, dtype="uint8")  # 初始化一个掩膜来存储大的斑点

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        orilabel = labelMask[150:400,220:500]
        numori = cv2.countNonZero(orilabel)    #初步选取一个块作为选区，选区外的不认

        # 对labelMask中的非零像素进行计数。如果numPixels超过了一个预先定义的阈值 ,那么认为这个斑点“足够大”，并将其添加到掩膜中
        numPixels = cv2.countNonZero(labelMask)
        if 250 < numPixels < 600 and numori != 0:
            mask = cv2.add(mask, labelMask)

    return mask


# for p in os.listdir(path):
#     original_img = cv2.imread(path + p)
#     gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     mask = highlight_point(gray_img)
#     if np.any(mask):
#         cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         cnts = contours.sort_contours(cnts)[0]
#         (x, y, w, h) = cv2.boundingRect(cnts[0])
#         ((cX, cY), radius) = cv2.minEnclosingCircle(cnts[0])
#         point = cX, cY
#     else:
#         point = cX, cY
#     with codecs.open('eyedata1/DSQ/point_no_fushi.txt', mode='a', encoding='utf-8') as file_txt:
#         file_txt.write(str(p) + ',' + str(cX) + ',' + str(cY) + '\n')
#     print('正在处理图片：', p)




label = getlabel('eyedata1/WLF/label.txt')
# point = getlabel('eyedata1/DSQ/point.txt')
point = getlabel('eyedata1/WLF/point_no_fushi.txt')

sum = 0
distance = []
ture = 0
false = 0
for i in range(len(label)):
    p1 = float(label[i][0]), float(label[i][1])
    p2 = float(point[i][0]), float(point[i][1])
    d = np.sqrt(np.square(p1[0] - p2[0])+np.square(p1[1] - p2[1]))
    sum = sum + d
    distance.append(d)
    if d < 8:
        ture = ture + 1
    else:
        false = false + 1


print('和为：', sum)
print('均值为：', sum/len(label))
print('最大误差：', max(distance))   # 7.184712944699374
print('正确个数为：', ture, '错误个数为：', false)
print('准确率为：', ture/(ture + false))


