#工具函数
import cv2
import numpy as np
from sklearn.cluster import KMeans

def autoshow(img):
    if img is None:
        print("图像为空，无法显示")
        return
    cv2.imshow("image", img)
    k = cv2.waitKey(0)  # 等待任意键
    if k & 0xFF == ord('q'):
        print("按下'q'退出")
    cv2.destroyAllWindows()
    return 

def denoise_image(img):
    return cv2.pyrMeanShiftFiltering(img, 10, 20)

def convert_color_space(img, space='YUV'):
    if space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'Lab':
        return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def cluster_colors(img, k=4):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(data)
    segmented = kmeans.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)
    return segmented

def binarize_image(img):
    y_channel = img[:, :, 0]
    _, binary = cv2.threshold(y_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_connected_components(binary_img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
    return num_labels, labels, stats, centroids

def morphological_refine(binary_img):
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed