#滤波与颜色转换
import os
import cv2
import numpy as np

IN_PUTIMAGE = "./data/DBY3.png"

image = cv2.imread(IN_PUTIMAGE)
print("原图像尺寸:", image.shape)
