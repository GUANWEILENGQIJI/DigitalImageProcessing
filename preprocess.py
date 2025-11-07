#滤波与颜色转换
import os
import cv2
import numpy as np

IN_PUTIMAGE = "./data/DBY3.png"

image = cv2.imread(IN_PUTIMAGE)
pixel = image[500, 800]
print(f"像素值:[B={pixel[0]}, G={pixel[1]}, R={pixel[2]}]")
