from skimage import io, measure
from skimage import data, color, draw, transform, feature, util
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from time import *
import collections
import imutils
from imutils import contours
from sklearn import preprocessing
import pandas as pd



path = 'eyedata1/FAM/'
name = 'video1/FAM5.mp4'
point = (349, 229)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 30
video = cv2.VideoWriter(name, fourcc, fps, (720, 576))
jishu = 0


def conv_cal(img, filter):
    h, w = img.shape
    img_filter = np.zeros([h, w])
    for i in range(h - 2):
        for j in range(w - 2):
            img_filter[i][j] = img[i][j] * filter[0][0] + img[i][j + 1] * filter[0][1] + img[i][j + 2] * filter[0][2] + \
                               img[i + 1][j] * filter[1][0] + img[i + 1][j + 1] * filter[1][1] + img[i + 1][j + 2] * \
                               filter[1][2] + \
                               img[i + 2][j] * filter[2][0] + img[i + 2][j + 1] * filter[2][1] + img[i + 2][j + 2] * \
                               filter[2][2]
    return img_filter


def draw_histogram(grayscale):
    gray_key = []
    gray_count = []
    gray_result = []
    histogram_gray = list(grayscale.ravel())  # 将多维数组转换成一维数组
    gray = dict(collections.Counter(histogram_gray))  # 统计图像中每个灰度级出现的次数
    gray = sorted(gray.items(), key=lambda item: item[0])  # 根据灰度级大小排序
    for element in gray:
        key = list(element)[0]
        count = list(element)[1]
        gray_key.append(key)
        gray_count.append(count)
    for i in range(0, 256):
        if i in gray_key:
            num = gray_key.index(i)
            gray_result.append(gray_count[num])
        else:
            gray_result.append(0)
    gray_result = np.array(gray_result)
    return gray_result


def histogram_equalization(histogram_e, lut_e, image_e):
    sum_temp = 0
    cf = []
    for i in histogram_e:
        sum_temp += i
        cf.append(sum_temp)
    for i, v in enumerate(lut_e):
        lut_e[i] = int(255.0 * (cf[i] / sum_temp) + 0.5)
    equalization_result = lut_e[image_e]
    return equalization_result


def erzhihua(img_t, t):
    w, h = img_t.shape
    for i in range(w):
        for j in range(h):
            if (img_t[i][j] > t):
                img_t[i][j] = 0
            else:
                img_t[i][j] = 255
    return img_t


def highlight_point(gray_img, pointt):
    blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)  # 然后将其转换为灰度图并进行平滑滤波，以减少高频噪声
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]  # 阈值化处理，p>=200，设置为255(白色)
    thresh = cv2.erode(thresh, None, iterations=2)  # 腐蚀
    thresh = cv2.dilate(thresh, None, iterations=4)  # 膨胀
    labels = measure.label(thresh, connectivity=2, background=0)  # 连接组件分析 label存储的为阈值图像每一斑点对应的正整数
    mask = np.zeros(thresh.shape, dtype="uint8")  # 初始化一个掩膜来存储大的斑点
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        # orilabel = labelMask[100:300, 220:500]
        orilabel = labelMask[350:550, 350:550]
        numori = cv2.countNonZero(orilabel)  # 初步选取一个块作为选区，选区外的不认
        # 对labelMask中的非零像素进行计数。如果numPixels超过了一个预先定义的阈值 ,那么认为这个斑点“足够大”，并将其添加到掩膜中
        numPixels = cv2.countNonZero(labelMask)
        if 250 < numPixels < 600 and numori != 0:
            mask = cv2.add(mask, labelMask)
    if np.any(mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        (point, radius) = cv2.minEnclosingCircle(cnts[0])
        point = (int(point[0]), int(point[1]))
        distance = np.sqrt((point[0] - pointt[0]) ** 2 + (point[1] - pointt[1]) ** 2)
        if distance > 50:
            point = pointt
    else:
        point = pointt

    return point


krisch1 = np.array([[1, 2, 1],
                    [-2, -4, -2],
                    [1, 2, 1]])
krisch2 = np.array([[1, -2, 1],
                    [2, -4, 2],
                    [1, -2, 1]])
krisch3 = np.array([[-2, 1, 2],
                    [1, -4, 1],
                    [2, 1, -2]])
krisch4 = np.array([[2, 1, -2],
                    [1, -4, 1],
                    [-2, 1, 2]])

begin_time = time()


for p in os.listdir(path):

    if p == '0001.jpg' or jishu == 4:

        original_img = cv2.imread(path + p)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        point = highlight_point(gray_img, point)
        if point[1] <= 200:
            gray_img = gray_img[1:401, point[0] - 200:point[0] + 200]
        else:
            gray_img = gray_img[point[1] - 200:point[1] + 200, point[0] - 200:point[0] + 200]
        gray_img = cv2.resize(gray_img, (200, 200))
        gray_img = cv2.resize(gray_img, (100, 100))

        w, h = gray_img.shape
        img = np.zeros([w + 2, h + 2])
        img[2:w + 2, 2:h + 2] = gray_img[0:w, 0:h]
        edge1 = conv_cal(img, krisch1)
        edge2 = conv_cal(img, krisch2)
        edge3 = conv_cal(img, krisch3)
        edge4 = conv_cal(img, krisch4)
        edge_img = np.zeros([w, h])
        for i in range(w):
            for j in range(h):
                edge_img[i][j] = max(list([abs(edge1[i][j]), abs(edge2[i][j]), abs(edge3[i][j]), abs(edge4[i][j])]))
                # if edge_img[i][j] > 254:
                #     edge_img[i][j] = 254

        min_max_scaler = preprocessing.MinMaxScaler()
        edge_img = min_max_scaler.fit_transform(edge_img) * 255  # 归一化
        # edge_img = erzhihua(edge_img,25)   # 二值化
        edge_img = edge_img.astype(np.uint8)  # 转换类型
        histogram = draw_histogram(edge_img)  # 直方图转化
        lut = np.zeros(256, dtype=edge_img.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
        image = histogram_equalization(histogram, lut, edge_img)  # 均衡化处理

        edges = feature.canny(image, sigma=3, low_threshold=10, high_threshold=50)  # 检测canny边缘

        hough_radii = np.arange(37, 38, 1)  # 半径范围
        hough_res = transform.hough_circle(edges, hough_radii)  # 圆变换
        centers = []  # 保存中心点坐标
        accums = []  # 累积值
        radii = []  # 半径
        for radius, h in zip(hough_radii, hough_res):
            # 每一个半径值，取出其中两个圆
            num_peaks = 1
            peaks = feature.peak_local_max(h, num_peaks=num_peaks)  # 取出峰值
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius] * num_peaks)

        center_x, center_y = centers[0]
        radius = radii[0]
        for b in range(7):
            cx, cy = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), radius * 4 -3+b)
            original_img[cy, cx] = (255, 0, 0)
        for b in range(7):
            cxs, cys = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), int(radius * 3 -3+b))
            original_img[cys, cxs] = (0, 0, 255)
        for b in range(7):
            cxb, cyb = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), int(radius * 5.3 -3+b))
            original_img[cyb, cxb] = (0, 255, 0)

        jishu = 0

    else:
        original_img = cv2.imread(path + p)
        center_x, center_y = centers[0]
        radius = radii[0]
        for b in range(7):
            cx, cy = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), radius * 4 -3+b)
            original_img[cy, cx] = (255, 0, 0)
        for b in range(7):
            cxs, cys = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), int(radius * 3 -3+b))
            original_img[cys, cxs] = (0, 0, 255)
        for b in range(7):
            cxb, cyb = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), int(radius * 5.3 -3+b))
            original_img[cyb, cxb] = (0, 255, 0)
        jishu = jishu + 1



    video.write(original_img)
    print('正在处理图片：', p)

end_time = time()
run_time = end_time - begin_time
print('程序运行的时间：', run_time)




