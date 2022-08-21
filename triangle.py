from skimage import io,color
import matplotlib.pyplot as plt
import numpy as np
import cv2
import collections
from sklearn import preprocessing
import pandas as pd

path = 'eyedata/DSQ/0001.jpg'
# path = 'eyedata/DSQ/yuan.png'

original_img=io.imread(path)
gray_img= cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_img,cmap="gray")
# plt.show()

# 将图片矩阵写入EXCEL
# excel = 'eyedata/DSQ/2.xlsx'
# data = pd.DataFrame(gray_img)
# writer = pd.ExcelWriter(excel)		# 写入Excel文件
# data.to_excel(writer, 'DSQ', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()


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


gray_img1 = cv2.resize(gray_img,(200,200))             # (200*200)
gray_img2 = cv2.resize(gray_img1,(100,100))            # (100*100)
gray_img3 = cv2.resize(gray_img2,(50,50))            # (50*50)


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

print(edge_img)
min_max_scaler = preprocessing.MinMaxScaler()
edge_img = min_max_scaler.fit_transform(edge_img) * 255   # 归一化
print(edge_img)
# edge_img = erzhihua(edge_img,25)   # 二值化
edge_img = edge_img.astype(np.uint8)  # 转换类型
print(edge_img)
histogram = draw_histogram(edge_img)  # 直方图转化
lut = np.zeros(256, dtype=edge_img.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
edge_img = histogram_equalization(histogram, lut, edge_img)  # 均衡化处理


w,h=gray_img1.shape
img=np.zeros([w+2,h+2])
img[2:w+2,2:h+2]=gray_img1[0:w,0:h]
edge1=conv_cal(img,krisch1)
edge2=conv_cal(img,krisch2)
edge3=conv_cal(img,krisch3)
edge4=conv_cal(img,krisch4)
edge_img1=np.zeros([w,h])
for i in range(w):
    for j in range(h):
        edge_img1[i][j]=max(list([abs(edge1[i][j]),abs(edge2[i][j]),abs(edge3[i][j]),abs(edge4[i][j])]))

min_max_scaler = preprocessing.MinMaxScaler()
edge_img1 = min_max_scaler.fit_transform(edge_img1) * 255   # 归一化
# edge_img1 = erzhihua(edge_img1,25)
edge_img1 = edge_img1.astype(np.uint8)
histogram = draw_histogram(edge_img1)  # 直方图转化
lut = np.zeros(256, dtype=edge_img1.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
edge_img1 = histogram_equalization(histogram, lut, edge_img1)  # 均衡化处理


w,h=gray_img2.shape
img=np.zeros([w+2,h+2])
img[2:w+2,2:h+2]=gray_img2[0:w,0:h]
edge1=conv_cal(img,krisch1)
edge2=conv_cal(img,krisch2)
edge3=conv_cal(img,krisch3)
edge4=conv_cal(img,krisch4)
edge_img2=np.zeros([w,h])
for i in range(w):
    for j in range(h):
        edge_img2[i][j]=max(list([abs(edge1[i][j]),abs(edge2[i][j]),abs(edge3[i][j]),abs(edge4[i][j])]))

min_max_scaler = preprocessing.MinMaxScaler()
edge_img2 = min_max_scaler.fit_transform(edge_img2) * 255   # 归一化
# edge_img2 = erzhihua(edge_img2,25)
edge_img2 = edge_img2.astype(np.uint8)
histogram = draw_histogram(edge_img2)  # 直方图转化
lut = np.zeros(256, dtype=edge_img2.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
edge_img2 = histogram_equalization(histogram, lut, edge_img2)  # 均衡化处理


w,h=gray_img3.shape
img=np.zeros([w+2,h+2])
img[2:w+2,2:h+2]=gray_img3[0:w,0:h]
edge1=conv_cal(img,krisch1)
edge2=conv_cal(img,krisch2)
edge3=conv_cal(img,krisch3)
edge4=conv_cal(img,krisch4)
edge_img3=np.zeros([w,h])
for i in range(w):
    for j in range(h):
        edge_img3[i][j]=max(list([abs(edge1[i][j]),abs(edge2[i][j]),abs(edge3[i][j]),abs(edge4[i][j])]))

min_max_scaler = preprocessing.MinMaxScaler()
edge_img3 = min_max_scaler.fit_transform(edge_img3) * 255   # 归一化
# edge_img3 = erzhihua(edge_img3,25)
edge_img3 = edge_img3.astype(np.uint8)
histogram = draw_histogram(edge_img3)  # 直方图转化
lut = np.zeros(256, dtype=edge_img3.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
edge_img3 = histogram_equalization(histogram, lut, edge_img3)  # 均衡化处理


plt.figure('imgs')
plt.subplot(221).set_title('edge_img')
plt.imshow(edge_img,cmap="gray")
plt.subplot(222).set_title('edge_img1')
plt.imshow(edge_img1,cmap="gray")
plt.subplot(223).set_title('edge_img2')
plt.imshow(edge_img2,cmap="gray")
plt.subplot(224).set_title('edge_img3')
plt.imshow(edge_img3,cmap="gray")
plt.show()


# 用霍夫变换检测圆
circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=40, minRadius=80, maxRadius=120)

# 整数化，#把circles包含的圆心和半径的值变成整数
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # if(i[1]>50 and i[1]<125 and i[0]<125 and i[0]>50):
        cv2.circle(gray_img, (i[0], i[1]), i[2], 255, 1)
        cv2.circle(gray_img, (i[0], i[1]), 2, 255, 1)


print("圆心坐标", i[0], i[1])
plt.subplot(121)
plt.imshow(gray_img)
plt.subplot(122)
plt.imshow(edge_img)
plt.show()
