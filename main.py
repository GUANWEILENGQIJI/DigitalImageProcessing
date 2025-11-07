import os
import cv2
import numpy as np
from utils import (
    denoise_image, autoshow,
    convert_color_space, cluster_colors,
    binarize_image, extract_connected_components,
    morphological_refine
)

#导入数据
IN_PUTDIR = "./data"

#总处理函数
def colordivide(image_dir):
    #读取图像
    for file_path in os.listdir(image_dir):
        if not file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        print(f"处理Processing {file_path}...")
        image_path = os.path.join(image_dir, file_path)
        image = cv2.imread(image_path)
        # 显示原图
        autoshow(image)
        #降噪（去背景点）
        image_denoise = denoise_image(image)
        autoshow(image_denoise)
        #颜色空间转换（分出亮度/色度）
        image_colorconv = convert_color_space(image_denoise,space='HSV')
        autoshow(image_colorconv)
        #聚类（红、蓝、黑、白分类）
        img_clustered, labels = cluster_colors(image_colorconv, k=4)
        autoshow(img_clustered)
        #连通域提取,阈值（二值化亮度区分文字）
        binary = binarize_image(img_clustered)
        components=extract_connected_components(binary)
        autoshow(binary)
    #形态学修正（连接断字）
        refined = morphological_refine(binary)
        autoshow(refined)
    #分层（每种颜色单独成图）
        for i in range(1, 4):  # 假设有3个前景颜色类别
            layer = np.zeros_like(image)
            layer[labels == i] = image[labels == i]
            autoshow(layer)
    #输出结果：


#test
colordivide(IN_PUTDIR)