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
IN_PUTIMAGE = "./R-C.jpg"
IN_PUTDIR = "./"

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

def remove_small_components(binary_img, min_area=30):
    """
    删除面积小于 min_area 的黑色前景连通域。
    """
    fg = cv2.bitwise_not(binary_img)  # 前景置为白
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros_like(fg)
    for i in range(1, num_labels):  # 跳过背景 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return cv2.bitwise_not(keep)

def erode_black(binary_img, kernel_size=3, iterations=1):
    """
    腐蚀暗色前景（黑色文字）。
    先反转为白前景再腐蚀，最后再反转回来。
    """
    inverted = cv2.bitwise_not(binary_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(inverted, kernel, iterations=iterations)
    return cv2.bitwise_not(eroded)

def sum_nonzero_pixels(img):
    """
    对图像中大于零的像素求和
    
    参数:
        img: 输入图像（可以是灰度图或彩色图）
    
    返回:
        灰度图: 返回标量（所有非零像素值的和）
        彩色图: 返回 (B_sum, G_sum, R_sum) 形状为 (3,) 的数组
    """
    if img.ndim == 2:  # 灰度图
        return np.sum(img[img > 0])
    else:  # 彩色图
        non_zero_mask = np.any(img > 0, axis=2)  # (H, W) 布尔掩码
        non_zero_pixels = img[non_zero_mask]  # (N, 3) 形状的数组
        if len(non_zero_pixels) > 0:
            return np.sum(non_zero_pixels, axis=0)  # 返回 (B_sum, G_sum, R_sum)
        else:
            return np.array([0, 0, 0], dtype=np.float64)

#对单个簇提取连通域
def single_cluster_binarize(cls,img,labels):
    mask = (labels == cls)                 # (H, W) 布尔掩码
    layer = np.zeros_like(img)           # 与原图同形状 dtype 的空图层   
    #直接用布尔掩码赋值
    layer[mask] = img[mask]
    #求质心颜色
    color=[]
    for c in range(3):
        channel_sum = sum_nonzero_pixels(layer[:, :, c])
        non_zero_count = np.count_nonzero(mask)
        if non_zero_count > 0:
            mean_color = channel_sum / non_zero_count
        else:
            mean_color = 0
        color.append(mean_color)
    color = np.array(color, dtype=np.uint8)  #转换为数组
    print(f"簇的编号: {cls}; 质心颜色: {color}")

    layer=np.ones_like(img)*255
    layer[mask]=img[mask]
    binary = binarize_image(layer)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_layers = morphological(binary, kernel_size=2, iterations=1)#连接断字
    closed_layers = remove_small_components(closed_layers, min_area=60)
    closed_layers = erode_black(closed_layers, kernel_size=3, iterations=1)#细化文字
    # colored_closed=np.zeros_like(img)
    mask_closed = (closed_layers == 0)
    closed_layers = np.ones_like(img) * 255
    closed_layers[mask_closed]=color
    return closed_layers
    # return layer
    # return binary
 

#处理单张图函数
def singleimage_process(file_path,image,space='YUV',k=3):
    #降噪锐化（去背景点）
    image_denoise = preprocess.process_image(image,'gaussian',factory=preprocess.gaussian_kernel,size=3,sigma=1.0)
    image_sharpen=preprocess.process_image(image_denoise,'sharpen',factory=preprocess.sharpen,size=3)
    #颜色空间转换（分出亮度/色度）
    image_cvt=convert_color_space(image_sharpen,space=space)
    # 先在亮度方向上聚类
    # labels,centroids=cluster_colors(image_cvt,k=4)
    labels,centroids=cluster_1dim(image_cvt[:,:,2],k=k)
    # print(centroids,labels.shape)
    #对每个簇连通域提取,阈值（二值化亮度区分文字）
    
    for cls in range(len(centroids)):
        centroids_uint8 = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
        colored_layer=single_cluster_binarize(cls,image_sharpen,labels)
        
        print(f"处理第{cls}簇...")
        # preprocess.autoshow(colored_layer)  

        colored_layer=np.array(colored_layer,dtype=np.uint8) 
        colored_layer_dn=preprocess.process_image(colored_layer,'gaussian',factory=preprocess.gaussian_kernel,size=3,sigma=1.0)
        colored_layer_en=preprocess.process_image(colored_layer_dn,'sharpen',factory=preprocess.sharpen,size=3)
        print(cv2.imwrite(f'E:/temp_code/python_Files/DigitalImageProcessing_CourseDesign/DigitalImageProcessing-main/data/result/Sample2/{file_path}slayer{cls}.jpg',colored_layer_en))

#总处理函数
def colordivide(image_dir,k):
    #读取图像
    for file_path in os.listdir(image_dir):
        if not file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        print(f"处理Processing {file_path}...")
        image_path = os.path.join(image_dir, file_path)
        image = cv2.imread(image_path)

        singleimage_process(file_path,image,k=k)


#test
colordivide('E:/temp_code/python_Files/DigitalImageProcessing_CourseDesign/DigitalImageProcessing-main/data/sample2/',k=3)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭所有窗口
plt.close('all')  # 关闭所有matplotlib窗口