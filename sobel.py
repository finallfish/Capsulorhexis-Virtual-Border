from skimage import io,color
import  matplotlib.pyplot as plt
import numpy as np

'''
original_img=data.chelsea()
gray_img=color.rgb2gray(original_img)


# #using system function
# edge_img=filters.roberts(gray_img)
# figure=plt.figure()
# plt.subplot(131).set_title('original_img')
# plt.imshow(original_img)
# plt.subplot(132).set_title('gray_img')
# plt.imshow(gray_img)
# plt.subplot(133).set_title('roberts_img')
# plt.imshow(edge_img)
# plt.show()


#self code
x_roberts=np.array([[1,0],
                    [0,-1]])
y_roberts=np.array([[0,1],
                    [-1,0]])
h,w=gray_img.shape
img=np.zeros([h+1,w+1])
img[1:h+1,1:w+1]=gray_img[0:h,0:w]
def robert_cal(img,filter):
    h,w=img.shape
    img_filter=np.zeros([h,w])
    for i in range(h-1):
        for j in range(w-1):
            img_filter[i][j]=img[i][j]*filter[0][0]+img[i][j+1]*filter[0][1]+img[i+1][j]*filter[1][0]+img[i+1][j+1]*filter[1][1]
    return img_filter

x_edge_img=robert_cal(img,x_roberts)
y_edge_img=robert_cal(img,y_roberts)
edge_img=np.zeros([h,w])
for i in range(h):
    for j in range(w):
      edge_img[i][j]=np.sqrt(x_edge_img[i][j]**2+y_edge_img[i][j]**2)/(np.sqrt(2))

plt.figure('imgs')
plt.subplot(321).set_title('original_img')
plt.imshow(original_img)
plt.subplot(322).set_title('gray_img')
plt.imshow(gray_img)
plt.subplot(323).set_title('x_edge_img')
plt.imshow(x_edge_img)
plt.subplot(324).set_title('y_edge_img')
plt.imshow(y_edge_img)
plt.subplot(325).set_title('edge_img')
plt.imshow(edge_img)

plt.show()
'''

'''
np.set_printoptions(threshold=100000000)

#input image
original_img=data.chelsea()
gray_img=color.rgb2gray(original_img)


# #using prewitt in package
# edge_img=filters.prewitt(gray_img)
# plt.figure('imgs')
# plt.subplot(131).set_title('original_img')
# plt.imshow(original_img)
# plt.subplot(132).set_title('gray_img')
# plt.imshow(gray_img)
# plt.subplot(133).set_title('edge_img')
# plt.imshow(edge_img)
# plt.show()



#self code
x_prewitt=np.array([[1,0,-1],
                   [1,0,-1],
                   [1,0,-1]])
y_prewitt=np.array([[1,1,1],
                   [0,0,0],
                   [-1,-1,-1]])
def conv_calculate(img,filter):
    h,w=img.shape
    conv_img=np.zeros([h-2,w-2])
    for i in range(h-2):
        for j in range(w-2):
            conv_img[i][j]=img[i][j]*filter[0][0]+img[i][j+1]*filter[0][1]+img[i][j+2]*filter[0][2]+\
                    img[i+1][j]*filter[1][0]+img[i+1][j+1]*filter[1][1]+img[i+1][j+2]*filter[1][2]+\
                    img[i+2][j]*filter[2][0]+img[i+2][j+1]*filter[2][1]+img[i+2][j+2]*filter[2][2]
    return conv_img

h,w=gray_img.shape
img=np.zeros([h+2,w+2])
img[2:h+2,2:w+2]=gray_img[0:h]
edge_x_img=conv_calculate(img,x_prewitt)
edge_y_img=conv_calculate(img,y_prewitt)

#p(i,j)=max[edge_x_img,edge_y_img]
edge_img_max=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        if edge_x_img[i][j]>edge_y_img[i][j]:
            edge_img_max[i][j]=edge_x_img[i][j]
        else:
            edge_img_max[i][j]=edge_y_img[i][j]

#p(i,j)=edge_x_img+edge_y_img
edge_img_sum=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        edge_img_sum[i][j]=edge_x_img[i][j]+edge_y_img[i][j]

# p(i,j)=|edge_x_img|+|edge_y_img|
edge_img_abs = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        edge_img_abs[i][j] = abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])


#p(i,j)=sqrt(edge_x_img**2+edge_y_img**2)
edge_img_sqrt=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        edge_img_sqrt[i][j]=np.sqrt((edge_x_img[i][j])**2+(edge_y_img[i][j])**2)

plt.figure('imgs')
plt.subplot(331).set_title('original_img')
plt.imshow(original_img)
plt.subplot(332).set_title('gray_img')
plt.imshow(gray_img)
plt.subplot(333).set_title('x_edge_img')
plt.imshow(edge_x_img)
plt.subplot(334).set_title('y_edge_img')
plt.imshow(edge_y_img)
plt.subplot(335).set_title('edge_img_max')
plt.imshow(edge_img_max)
plt.subplot(336).set_title('edge_img_sum')
plt.imshow(edge_img_sum)
plt.subplot(337).set_title('edge_img_sqrt')
plt.imshow(edge_img_sqrt)
plt.subplot(338).set_title('edge_img_abs')
plt.imshow(edge_img_abs)
plt.show()
'''

np.set_printoptions(threshold=100000000)
path = 'eyedata/DSQ/0001.jpg'
original_img = io.imread(path)
gray_img=color.rgb2gray(original_img)

# edge_img=filters.sobel(gray_img)
def sobel_cal(img,filter):
    h,w=img.shape
    img_filter=np.zeros([h,w])
    for i in range(h-2):
        for j in range(w-2):
            img_filter[i][j]=img[i][j]*filter[0][0]+img[i][j+1]*filter[0][1]+img[i][j+2]*filter[0][2]+\
                    img[i+1][j]*filter[1][0]+img[i+1][j+1]*filter[1][1]+img[i+1][j+2]*filter[1][2]+\
                    img[i+2][j]*filter[2][0]+img[i+2][j+1]*filter[2][1]+img[i+2][j+2]*filter[2][2]
    return img_filter


# x_sobel=np.array([[-1,0,1],
#                   [-2,0,2],
#                   [-1,0,1]])
# y_sobel=np.array([[1,2,1],
#                   [0,0,0],
#                   [-1,-2,-1]])

# x_sobel=np.array([[2,1,0],
#                   [1,0,-1],
#                   [0,-1,-2]])
# y_sobel=np.array([[0,1,2],
#                   [-1,0,1],
#                   [-2,-1,0]])

x_sobel=np.array([[-2,-1,0],
                  [-1,-1,0],
                  [0,0,5]])
y_sobel=np.array([[0,0,5],
                  [-1,-1,0],
                  [-2,-1,0]])

h,w=gray_img.shape
img=np.zeros([h+2,w+2])
img[2:h+2,2:w+2]=gray_img[0:h,0:w]
edge_x_img=sobel_cal(img,x_sobel)
edge_y_img=sobel_cal(img,y_sobel)


edge_img_max=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        if edge_x_img[i][j]>edge_y_img[i][j]:
            edge_img_max[i][j]=edge_x_img[i][j]
        else:
            edge_img_max[i][j]=edge_y_img[i][j]


edge_img_sum=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        edge_img_sum[i][j]=edge_x_img[i][j]+edge_y_img[i][j]


edge_img_abs = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        edge_img_abs[i][j] = abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])


edge_img_sqrt=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        edge_img_sqrt[i][j]=np.sqrt((edge_x_img[i][j])**2+(edge_y_img[i][j])**2)


edge_img_01=np.zeros([h,w])
for i in range(h):
    for j in range(w):
        if (((abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])) > 100/255) and
                ((abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])) < 200/255)):
            edge_img_01[i][j]=0
        else:
            edge_img_01[i][j]=255

plt.figure('imgs')
plt.subplot(331).set_title('original_img')
plt.imshow(original_img)
plt.subplot(332).set_title('gray_img')
plt.imshow(gray_img,cmap="gray")
plt.subplot(333).set_title('x_edge_img')
plt.imshow(edge_x_img,cmap="gray")
plt.subplot(334).set_title('y_edge_img')
plt.imshow(edge_y_img,cmap="gray")
plt.subplot(335).set_title('edge_img_max')
plt.imshow(edge_img_max,cmap="gray")
plt.subplot(336).set_title('edge_img_sum')
plt.imshow(edge_img_sum,cmap="gray")
plt.subplot(337).set_title('edge_img_sqrt')
plt.imshow(edge_img_sqrt,cmap="gray")
plt.subplot(338).set_title('edge_img_abs')
plt.imshow(edge_img_abs,cmap="gray")
plt.subplot(339).set_title('edge_img_01')
plt.imshow(edge_img_01,cmap="gray")
plt.show()


# edge_img_01=np.zeros([h,w])
# for k in range(255):
#     # for l in range(k+1,k+1):
#         temp = 0
#         for i in range(h):
#             for j in range(w):
#                 judge = abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])
#                 if (judge > k/255 and judge < (k+1)/255):
#                     temp = temp +1
#         print('当阈值为',k+1,'到',k+2,'时有',temp,'个特征点')



