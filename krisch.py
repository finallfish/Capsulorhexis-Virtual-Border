from skimage import io,color
import matplotlib.pyplot as plt
import numpy as np

path = 'eyedata/DSQ/0001.jpg'
original_img=io.imread(path)
gray_img=color.rgb2gray(original_img)

def conv_cal(img,filter):
    h,w=img.shape
    img_filter=np.zeros([h,w])
    for i in range(h-2):
        for j in range(w-2):
            img_filter[i][j]=img[i][j]*filter[0][0]+img[i][j+1]*filter[0][1]+img[i][j+2]*filter[0][2]+\
                    img[i+1][j]*filter[1][0]+img[i+1][j+1]*filter[1][1]+img[i+1][j+2]*filter[1][2]+\
                    img[i+2][j]*filter[2][0]+img[i+2][j+1]*filter[2][1]+img[i+2][j+2]*filter[2][2]
    return img_filter

krisch1=np.array([[5,5,5],
                  [-3,0,-3],
                  [-3,-3,-3]])
krisch2=np.array([[-3,-3,-3],
                  [-3,0,-3],
                  [5,5,5]])
krisch3=np.array([[5,-3,-3],
                  [5,0,-3],
                  [5,-3,-3]])
krisch4=np.array([[-3,-3,5],
                  [-3,0,5],
                  [-3,-3,5]])
krisch5=np.array([[-3,-3,-3],
                  [-3,0,5],
                  [-3,5,5]])
krisch6=np.array([[-3,-3,-3],
                  [5,0,-3],
                  [5,5,-3]])
krisch7=np.array([[-3,5,5],
                  [-3,0,5],
                  [-3,-3,-3]])
krisch8=np.array([[5,5,-3],
                  [5,0,-3],
                  [-3,-3,-3]])

w,h=gray_img.shape
img=np.zeros([w+2,h+2])
img[2:w+2,2:h+2]=gray_img[0:w,0:h]
edge1=conv_cal(img,krisch1)
edge2=conv_cal(img,krisch2)
edge3=conv_cal(img,krisch3)
edge4=conv_cal(img,krisch4)
edge5=conv_cal(img,krisch5)
edge6=conv_cal(img,krisch6)
edge7=conv_cal(img,krisch7)
edge8=conv_cal(img,krisch8)
edge_img=np.zeros([w,h])
for i in range(w):
    for j in range(h):
        edge_img[i][j]=max(list([edge1[i][j],edge2[i][j],edge3[i][j],edge4[i][j],edge5[i][j],edge6[i][j],edge7[i][j],edge8[i][j]]))
for i in range(w):
    for j in range(h):
        if(edge_img[i][j]> 150/255):
            edge_img[i][j]= 0
        else:
            edge_img[i][j] = 255


plt.figure('imgs')
plt.subplot(331).set_title('edge1')
plt.imshow(edge1,cmap="gray")
plt.subplot(332).set_title('edge2')
plt.imshow(edge2,cmap="gray")
plt.subplot(333).set_title('edge3')
plt.imshow(edge3,cmap="gray")
plt.subplot(334).set_title('edge4')
plt.imshow(edge4,cmap="gray")
plt.subplot(335).set_title('edge5')
plt.imshow(edge5,cmap="gray")
plt.subplot(336).set_title('edge6')
plt.imshow(edge6,cmap="gray")
plt.subplot(337).set_title('edge7')
plt.imshow(edge7,cmap="gray")
plt.subplot(338).set_title('edge8')
plt.imshow(edge8,cmap="gray")
plt.subplot(339).set_title('edge')
plt.imshow(edge_img,cmap="gray")
plt.show()
