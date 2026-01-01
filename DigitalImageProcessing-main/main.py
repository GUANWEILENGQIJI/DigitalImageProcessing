import os
import cv2
import numpy as np
import utils
import preprocess 
""" from utils import (
    denoise_image,
    convert_color_space, cluster_colors,
    binarize_image, extract_connected_components,
    morphological_refine
) """
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

#导入数据
IN_PUTIMAGE = "./data/DBY3.png"
IN_PUTDIR = "./data"

def convert_color_space(img, space='YUV'):
    if space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'Lab':
        return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

#三维图像聚类函数
def cluster_colors(img, k=3):
    H, W = img.shape[:2]
    # 将图像展开为二维数组 (N, 3)
    data = img.reshape((-1, 3))
    data = np.float32(data)
    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    #获取类标签
    labels=kmeans.labels_
    #获取质心
    centroids=kmeans.cluster_centers_
    # plt.scatter(data[:,0],data[:,1],data[:,2],c=labels,cmap='jet')
    # plt.show()
    return  labels.reshape((H,W)),centroids

#对图像单一维度聚类图像
def cluster_1dim(_img,k=2):
    data=_img.reshape((-1,1))
    data = np.float32(data)
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    #获取类标签
    labels=kmeans.labels_
    #获取质心
    centroids=kmeans.cluster_centers_
    H, W = _img.shape
    return  labels.reshape((H,W)),centroids

def binarize_image(img):
    y_channel = img[:, :, 0]
    _, binary = cv2.threshold(y_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
def morphological(binary_img, kernel_size=5, iterations=2):
    """
    膨胀暗色区域（黑色文字），用于连接断裂的笔画
    
    参数:
        binary_img: 二值图像 (黑色前景, 白色背景)
        kernel_size: 膨胀核的大小
        iterations: 膨胀迭代次数
    
    返回:
        膨胀后的二值图像
    """
    # 反转二值图：黑色变白色，白色变黑色
    inverted = cv2.bitwise_not(binary_img)
    # 对反转后的图像进行膨胀（白色膨胀）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(inverted, kernel, iterations=iterations)
    # 再次反转回来：恢复原始色系，但黑色已经膨胀
    result = cv2.bitwise_not(dilated)
    return result


#对单个簇提取连通域
def single_cluster_binarize(cls,img,labels,centroids):
    mask = (labels == cls)                 # (H, W) 布尔掩码
    layer = np.zeros_like(img)           # 与原图同形状 dtype 的空图层   
    color = centroids[cls].astype(np.uint8)                 # 取出该簇的质心颜色（三通道）
    #直接用布尔掩码赋值
    layer[mask] = img[mask]
    color=cv2.mean(layer)[:3]
    color = np.array(color, dtype=np.uint8)  # ← 转换为数组
    print(f"簇的编号: {cls}; 质心颜色: {color}")

    layer=np.ones_like(img)*255
    layer[mask]=img[mask]
    binary = binarize_image(layer)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_layers = morphological(binary, kernel_size=3, iterations=1)#连接断字
    # colored_closed=np.zeros_like(img)
    mask_closed = (closed_layers == 0)
    closed_layers = np.ones_like(img) * 255
    closed_layers[mask_closed]=color
    return closed_layers
    # return layer
    # return binary
 

#处理单张图函数
def singleimage_process(image,space='YUV'):
    #降噪锐化（去背景点）
    image_denoise = preprocess.process_image(image,'gaussian',factory=preprocess.gaussian_kernel,size=3,sigma=1.0)
    image_sharpen=preprocess.process_image(image,'sharpen',factory=preprocess.sharpen,size=3)
    #颜色空间转换（分出亮度/色度）
    image_cvt=convert_color_space(image_sharpen,space=space)
    # 先在亮度方向上聚类
    # labels,centroids=cluster_colors(image_cvt,k=4)
    labels,centroids=cluster_1dim(image_cvt[:,:,2])
    # print(centroids,labels.shape)
    #对每个簇连通域提取,阈值（二值化亮度区分文字）
    closed_layers=[]
    for cls in range(len(centroids)):
        centroids_uint8 = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
        colored_layer=single_cluster_binarize(cls,image_sharpen,labels,centroids_uint8)
        closed_layers.append(colored_layer)
        print(f"处理第{cls}簇...")
        preprocess.autoshow(closed_layers[-1])    
    

    image_final = np.ones_like(image)*255
    
    return image_final

#总处理函数
def colordivide(image_dir):
    #读取图像
    for file_path in os.listdir(image_dir):
        if not file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        print(f"处理Processing {file_path}...")
        image_path = os.path.join(image_dir, file_path)
        image = cv2.imread(image_path)

        image_final=singleimage_process(image)
        # preprocess.autoshow(image_final)
        # utils.show_y_hist(image_final, bins=256, normalize=True, save_path=None, figsize=(6,4))
        img_to_save = image_final
        img_to_save=np.clip(img_to_save, 0, 255).astype(np.uint8)
        print(cv2.imwrite('./data/result/result1.jpg',img_to_save))

""" 
        
        #连通域提取,阈值（二值化亮度区分文字）
        binary = binarize_image(img_clustered)
        components=extract_connected_components(binary)
        # autoshow(binary)
    #形态学修正（连接断字）
        refined = morphological_refine(binary)
        # autoshow(refined)
    #分层（每种颜色单独成图）
        for i in range(1, 4):  # 假设有3个前景颜色类别
            layer = np.zeros_like(image)
            layer[labels == i] = image[labels == i]
            autoshow(layer)
    #输出结果：
 """

#test
colordivide(IN_PUTDIR) 