from skimage import measure
from skimage import data, color,draw,transform,feature,util
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

path = 'eyedata/WLF/'
name = 'video/HighLight-WLF.mp4'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 30
video = cv2.VideoWriter(name, fourcc, fps, (720, 576))

def highlight_point(gray_img):
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

        orilabel = labelMask[100:300,220:500]
        numori = cv2.countNonZero(orilabel)    #初步选取一个块作为选区，选区外的不认

        # 对labelMask中的非零像素进行计数。如果numPixels超过了一个预先定义的阈值 ,那么认为这个斑点“足够大”，并将其添加到掩膜中
        numPixels = cv2.countNonZero(labelMask)
        if 250 < numPixels < 600 and numori != 0:
            mask = cv2.add(mask, labelMask)

    return mask


for p in os.listdir(path):

    original_img = cv2.imread(path + p)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    mask = highlight_point(gray_img)

    if np.any(mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]

        (x, y, w, h) = cv2.boundingRect(cnts[0])
        ((cX, cY), radius) = cv2.minEnclosingCircle(cnts[0])
        cv2.circle(original_img, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)

    else:
        cv2.circle(original_img, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)

    video.write(original_img)
    print('正在处理图片：', p)