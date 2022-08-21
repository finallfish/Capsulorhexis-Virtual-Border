import math
import numpy as np
from timebudget import timebudget

iterations_count = round(1e7)

def complex_operation(input_index):
    print("Complex operation. Input index: {:2d}".format(input_index))
    [math.exp(i) * math.sinh(i) for i in [1] * iterations_count]


# @timebudget
def run_complex_operations(operation, input):
    for i in input:
        operation(i)

input = range(10)
run_complex_operations(complex_operation, input)




'''
测试程序
from numpy import random
import numpy as np
from time import *
import pandas as pd
from joblib import Parallel, delayed

def xunhuan(img, img_filter, i):
    for j in range(100):
        img_filter[i][j] = img[i][j]+(i+1)+(j+1)

    return img_filter



# img = random.randint(10, 100, size=(5,5))
img = np.zeros([100,100])
img_filter = np.zeros([100,100])

img_filter = Parallel(n_jobs=16)(delayed(xunhuan)(img,img_filter,i) for i in range(100))
img_filter = np.array(img_filter)
print(img_filter.ndim)

begin_time = time()
t = img_filter.sum(axis=0)
end_time = time()

# for i in range (5):
#     for j in range(5):
#         img_filter[i][j] = img[i][j]+(i+1)+(j+1)
print(t.shape)

run_time = end_time - begin_time
print('程序运行的时间：', run_time)
'''
