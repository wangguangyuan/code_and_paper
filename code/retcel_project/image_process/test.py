# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:24:42 2017

@author: wgy
"""


import matplotlib.pyplot as plt
from skimage import data,color,morphology,feature
from scipy import ndimage


#生成二值测试图像
img=color.rgb2gray(data.coins())
#检测canny边缘,得到二值图片
edgs=feature.canny(img, sigma=3, low_threshold=10, high_threshold=50) 

chull = morphology.convex_hull_object(edgs)

#绘制轮廓
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1= axes.ravel()
ax0.imshow(edgs,plt.cm.gray)
ax0.set_title('many objects')
ax1.imshow(chull,plt.cm.gray)
ax1.set_title('convex_hull image')
plt.show()