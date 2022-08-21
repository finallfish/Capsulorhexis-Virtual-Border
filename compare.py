from skimage import io,color
import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn import preprocessing
from skimage.measure import compare_ssim, compare_psnr, compare_mse
from skimage import data, color,draw,transform,feature,util
import os
import cv2
import codecs
import imutils
from imutils import contours
from skimage import io, measure
import warnings
warnings.filterwarnings("ignore")

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
        orilabel = labelMask[100:300,220:500]
        numori = cv2.countNonZero(orilabel)    #初步选取一个块作为选区，选区外的不认
        # 对labelMask中的非零像素进行计数。如果numPixels超过了一个预先定义的阈值 ,那么认为这个斑点“足够大”，并将其添加到掩膜中
        numPixels = cv2.countNonZero(labelMask)
        if 250 < numPixels < 600 and numori != 0:
            mask = cv2.add(mask, labelMask)
    if np.any(mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        (point, radius) = cv2.minEnclosingCircle(cnts[0])
        point = (int(point[0]),int(point[1]))
        distance = np.sqrt((point[0] - pointt[0])**2 + (point[1] - pointt[1])**2)
        if distance > 50:
            point = pointt
    else:
        point = pointt

    return point

def double_circle(im,point,r2,r1):
    # im为图像，point为圆心坐标，r2为大半径，r1为小半径
    c = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.uint8)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            center = np.array([int(point[0]), int(point[1])])
            t = np.array([i, j])
            if r1 < (sum((t - center) ** 2)) ** (1 / 2) < r2:
                c[i, j] = im[i, j]
            else:
                c[i, j] = 0
    return c

def junhenghua(a_image):
    min_max_scaler = preprocessing.MinMaxScaler()
    b_image = min_max_scaler.fit_transform(a_image) * 255  # 归一化
    # edge_img = erzhihua(edge_img,25)   # 二值化
    b_image = b_image.astype(np.uint8)  # 转换类型
    histogram = draw_histogram(b_image)  # 直方图转化
    lut = np.zeros(256, dtype=b_image.dtype)  # 创建空的查找表,返回image类型的用0填充的数组；
    b_image = histogram_equalization(histogram, lut, b_image)  # 均衡化处理

    return b_image

def caculater(number,gray_temp,result_temp,imaged_temp):
    psnr = compare_psnr(gray_temp, result_temp)
    ssim = compare_ssim(gray_temp, result_temp)
    mse = compare_mse(gray_temp, result_temp)
    psnrresult[number] += psnr
    ssimresult[number] += ssim
    mseresult[number] += mse

    with codecs.open(savepath, mode='a', encoding='utf-8') as file_txt:
        file_txt.write(str(mse) + ',' + str(ssim) + ',' + str(psnr) + ',')

    psnr = compare_psnr(gray_temp, imaged_temp)
    ssim = compare_ssim(gray_temp, imaged_temp)
    mse = compare_mse(gray_temp, imaged_temp)
    psnredge[number] += psnr
    ssimedge[number] += ssim
    mseedge[number] += mse

    with codecs.open(savepath, mode='a', encoding='utf-8') as file_txt:
        file_txt.write(str(mse) + ',' + str(ssim) + ',' + str(psnr) + ',')


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


path = 'eyedata1/DSQ/'
savepath = 'result-image/DSQ.txt'
point = (349,229)
count = 0
false = 0
psnrresult = np.zeros((5,1))
ssimresult = np.zeros((5,1))
mseresult = np.zeros((5,1))
psnredge = np.zeros((5,1))
ssimedge = np.zeros((5,1))
mseedge = np.zeros((5,1))
name = ('Sobel', 'Scharry', 'Laplaian', 'Canny', 'Ridge')


for p in os.listdir(path):

    print('正在执行图像：', p)
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

    image = junhenghua(edge_img)  # 均衡化处理

    edges = feature.canny(image, sigma=3, low_threshold=10, high_threshold=50)  # 检测canny边缘

    hough_radii = np.arange(31, 32, 1)  # 半径范围
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
    cx, cy = draw.circle_perimeter(center_y * 4 +(point[0]-200), center_x*4+(point[1]-200), radius * 4)
    original_img[cy, cx] = (255, 0, 0)
    cxs, cys = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), int(radius * 3))
    original_img[cys, cxs] = (0, 0, 255)
    cxb, cyb = draw.circle_perimeter(center_y * 4 + (point[0] - 200), center_x * 4 + (point[1] - 200), int(radius * 5.3))
    original_img[cyb, cxb] = (0, 255, 0)

    cv2.imshow(p, original_img)
    key = cv2.waitKey()

    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(gray_img,cmap = 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(result, cmap = 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(imaged, cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(original_img)
    # plt.show()

    if key == 48:

        print('成功检测的图像，纳入统计中')

        with codecs.open(savepath, mode='a', encoding='utf-8') as file_txt:
            file_txt.write(str(p) + ',')

        gray_img_circle = double_circle(gray_img, centers[0], radii[0] * 1.25, radii[0] * 0.75)
        result_ridge = double_circle(edge_img, centers[0], radii[0] * 1.25, radii[0] * 0.75)
        imaged_ridge = double_circle(image, centers[0], radii[0] * 1.25, radii[0] * 0.75)
        caculater(4, gray_img_circle, result_ridge, imaged_ridge)

        # # Sobel边缘检测算子
        x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
        # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
        # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        result_sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 提取的特征图
        image_sobel = junhenghua(result_sobel)  # 灰度均衡化
        result_sobel = double_circle(result_sobel, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 特征图的取环
        image_sobel = double_circle(image_sobel, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 均衡化图取环
        caculater(0, gray_img_circle, result_sobel, image_sobel)

        # Scharry算子
        x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=-1)
        y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=-1)
        Scharr_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scharr_absY = cv2.convertScaleAbs(y)
        result_scharry = cv2.addWeighted(Scharr_absX, 0.5, Scharr_absY, 0.5, 0)
        image_scharry = junhenghua(result_scharry)  # 灰度均衡化
        result_scharry = double_circle(result_scharry, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 特征图的取环
        image_scharry = double_circle(image_scharry, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 均衡化图取环
        caculater(1, gray_img_circle, result_scharry, image_scharry)

        # 拉普拉斯算子
        blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_16S, ksize=5)
        result_lap = cv2.convertScaleAbs(laplacian)
        image_lap = junhenghua(result_lap)  # 灰度均衡化
        result_lap = double_circle(result_lap, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 特征图的取环
        image_lap = double_circle(image_lap, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 均衡化图取环
        caculater(2, gray_img_circle, result_lap, image_lap)

        # canny算子
        blur = cv2.GaussianBlur(gray_img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
        result_canny = cv2.Canny(blur, 0, 100)  # 50是最小阈值,150是最大阈值
        image_canny = junhenghua(result_canny)  # 灰度均衡化
        result_canny = double_circle(result_canny, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 特征图的取环
        image_canny = double_circle(image_canny, centers[0], radii[0] * 1.25, radii[0] * 0.75)  # 均衡化图取环
        caculater(3, gray_img_circle, result_canny, image_canny)

        with codecs.open(savepath, mode='a', encoding='utf-8') as file_txt:
            file_txt.write('\n')

        count += 1
        cv2.destroyAllWindows()
        print('正确结果个数：', count, '错误结果个数：', false)
        print(name[0],':',psnredge[0]/count,'，',name[1],':',psnredge[1]/count,
              '，',name[2],':',psnredge[2]/count,'，',name[3],':',psnredge[3]/count,
              '，',name[4],':',psnredge[4]/count)


    else:
        print('未成功检测的图像，丢弃')
        false += 1
        cv2.destroyAllWindows()
        print('正确结果个数：', count, '错误结果个数：', false)


'''输出结果'''
for hao in range(5):
    print('当前输出的为',name[hao],'算子统计结果')
    print('mseresult', mseresult[hao] / count)
    print('ssimresult', ssimresult[hao] / count)
    print('psnrresult', psnrresult[hao] / count)
    print('mseedge', mseedge[hao] / count)
    print('ssimedge', ssimedge[hao] / count)
    print('psnredge', psnredge[hao] / count)







