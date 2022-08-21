from skimage import data, color,draw,transform,feature,util
from skimage import io,color
import matplotlib.pyplot as plt
import numpy as np
import cv2
import collections
from sklearn import preprocessing
import pandas as pd

path = 'eyedata/DSQ/0002.jpg'

original_img=io.imread(path)
gray_img= cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.resize(gray_img,(200,200))
gray_img = cv2.resize(gray_img,(100,100))



def conv_cal(img,filter):
    h,w=img.shape
    img_filter=np.zeros([h,w])
    for i in range(h-2):
        for j in range(w-2):
            img_filter[i][j]=img[i][j]*filter[0][0]+img[i][j+1]*filter[0][1]+img[i][j+2]*filter[0][2]+\
                    img[i+1][j]*filter[1][0]+img[i+1][j+1]*filter[1][1]+img[i+1][j+2]*filter[1][2]+\
                    img[i+2][j]*filter[2][0]+img[i+2][j+1]*filter[2][1]+img[i+2][j+2]*filter[2][2]
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


def erzhihua(img_t,t):

    w, h = img_t.shape
    for i in range(w):
        for j in range(h):
            if (img_t[i][j] > t):
                img_t[i][j] = 0
            else:
                img_t[i][j] = 255
    return img_t


krisch1=np.array([[1,2,1],
                  [-2,-4,-2],
                  [1,2,1]])
krisch2=np.array([[1,-2,1],
                  [2,-4,2],
                  [1,-2,1]])
krisch3=np.array([[-2,1,2],
                  [1,-4,1],
                  [2,1,-2]])
krisch4=np.array([[2,1,-2],
                  [1,-4,1],
                  [-2,1,2]])


w,h=gray_img.shape
img=np.zeros([w+2,h+2])
img[2:w+2,2:h+2]=gray_img[0:w,0:h]
edge1=conv_cal(img,krisch1)
edge2=conv_cal(img,krisch2)
edge3=conv_cal(img,krisch3)
edge4=conv_cal(img,krisch4)
edge_img=np.zeros([w,h])
for i in range(w):
    for j in range(h):
        edge_img[i][j]=max(list([abs(edge1[i][j]),abs(edge2[i][j]),abs(edge3[i][j]),abs(edge4[i][j])]))
        # if edge_img[i][j] > 254:
        #     edge_img[i][j] = 254

min_max_scaler = preprocessing.MinMaxScaler()
edge_img = min_max_scaler.fit_transform(edge_img) * 255   # 归一化
# edge_img = erzhihua(edge_img,25)   # 二值化
edge_img = edge_img.astype(np.uint8)  # 转换类型
histogram = draw_histogram(edge_img)  # 直方图转化
lut = np.zeros(256, dtype=edge_img.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
image = histogram_equalization(histogram, lut, edge_img)  # 均衡化处理


edges =feature.canny(image, sigma=3, low_threshold=10, high_threshold=50)   # 检测canny边缘


fig, (ax0,ax1) = plt.subplots(1,2, figsize=(8, 5))

ax0.imshow(edges, cmap='gray')    # 显示canny边缘
ax0.set_title('original iamge')


hough_radii = np.arange(32, 33 ,1)   # 半径范围

hough_res = transform.hough_circle(edges, hough_radii)   # 圆变换

centers = []   # 保存中心点坐标
accums = []   # 累积值
radii = []   # 半径

for radius, h in zip(hough_radii, hough_res):
 # 每一个半径值，取出其中两个圆
 num_peaks = 1     # 选取峰值最大的一个圆
 peaks =feature.peak_local_max(h, num_peaks=num_peaks) #取出峰值
 centers.extend(peaks)
 accums.extend(h[peaks[:, 0], peaks[:, 1]])
 radii.extend([radius] * num_peaks)


# #画出最接近的5个圆
# # image = color.gray2rgb(image)
# for idx in np.argsort(accums)[::-1][:5]:
#     print(idx)
#     center_x, center_y = centers[idx]
#     radius = radii[idx]
#     cx, cy =draw.circle_perimeter(center_y*4, center_x*4, radius*4)
#     original_img[cy, cx] = (255,0,0)

center_x, center_y = centers[0]
radius = radii[0]
cx, cy =draw.circle_perimeter(center_y*4, center_x*4, radius*4)
original_img[cy, cx] = (255,0,0)


ax1.imshow(original_img)
ax1.set_title('detected image')
plt.show()