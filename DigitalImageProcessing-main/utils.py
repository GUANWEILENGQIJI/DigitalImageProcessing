#工具函数
import cv2
import numpy as np
import matplotlib.pyplot as plt

IN_PUTIMAGE = "./data/DBY3.png"

def show_y_hist(img_bgr, bins=256, normalize=False, save_path=None, figsize=(6,4)):
    """
    计算并绘制图像的 YUV 亮度通道直方图。
    - img_bgr: BGR 格式的图像（cv2.imread 的结果）
    - bins: 直方图 bin 数
    - normalize: 是否归一化为概率（sum=1）
    - save_path: 若不为 None，则保存图像到该路径
    """
    import matplotlib.pyplot as plt
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    hist = cv2.calcHist([y], [0], None, [bins], [0, 256]).ravel()
    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total
    x = np.arange(bins)
    plt.figure(figsize=figsize)
    plt.plot(x, hist, color='k')
    plt.fill_between(x, hist, color='gray', alpha=0.3)
    plt.xlim([0, bins-1])
    plt.xlabel('Y intensity (0-255)')
    plt.ylabel('Probability' if normalize else 'Count')
    plt.title('Y (luma) histogram (YUV)')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
""" 
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

 """