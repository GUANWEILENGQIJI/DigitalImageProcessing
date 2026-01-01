#滤波与颜色转换
import cv2
import json
import numpy as np

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

#定义卷积核
class KernelBank:
    """
    支持注册固定核或核生成器(factory)
    用法:
      kb = KernelBank()
      kb.register("sharpen", kernel=np.array(...))
      kb.register("gauss", factory=my_gauss_factory)
      k = kb.get("gauss", size=5, sigma=1.2)
      out = kb.apply(img, "gauss", size=5, sigma=1.2, per_channel=False)
    """
    def __init__(self):
        self.kernels={}
    
    def register(self,name,kernel=None,factory=None):
        key=str(name)
        entry={'kernel':None,'factory':None}
        if kernel is not None:
            entry['kernel']=np.asarray(kernel,dtype=np.float32)
        if factory is not None:
            if not callable(factory):
                raise TypeError("Factory must be callable")
            entry['factory']=factory
        self.kernels[key]=entry

    def set_kernel(self,name,kernel):
        key=str(name)
        if key not in self.kernels:
            self.kernels[key]={'kernel':None,'factory':None}
        self.kernels[key]=np.asarray(kernel,dtype=np.float32)

    def set_factory(self,name,factory):
        """
        获取核：
        - 若注册了固定 kernel 则直接返回（不管是否传参）；
        - 否则若注册了 factory 则每次调用 factory(*args, **kwargs) 生成并返回（不缓存）。
        """
        if factory is not callable:
            raise TypeError("Factory must be callable")
        key=str(name)
        if key not in self.kernels:
            self.kernels[key]={'kernel':None,'factory':None}
        self.kernels[key]['factory']=factory

    def get_kernel(self,name,*args,**kwargs):
        key=str(name)
        if name not in self.kernels:
            raise TypeError(f"{name} not exits")
        entry=self.kernels[key]
        if entry['kernel'] is not None:
            return entry['kernel']
        if entry['factory'] is None:
            raise TypeError(f"'{name}' has no kernel or factory")
        kernel=np.asarray(entry['factory'](*args,**kwargs),dtype=np.float32)
        if kernel.ndim !=2:
            raise TypeError("generate kernel must be 2D!")
        return kernel

    def apply(self,img,name,*args,per_channel=True,borderType=cv2.BORDER_REFLECT,**kwargs):
        """
        将指定核应用到图像并返回结果（uint8）。
        - img: HxW 或 HxWxC numpy ndarray
        - name: 注册名
        - *args/**kwargs: 若使用 factory 则传给 factory
        - per_channel: True 对每个通道独立应用；False 仅对亮度通道(Y)应用
        """
        kernel=self.get_kernel(name,*args,**kwargs)
        if img is None:
            return None
        if img.ndim==2:
            out=cv2.filter2D(img,ddepth=-1,kernel=kernel,borderType=borderType)
            return np.clip(out,0,255).astype(np.uint8)
        if per_channel:
            out=cv2.filter2D(img,ddepth=-1,kernel=kernel,borderType=borderType)
            return np.clip(out,0,255).astype(np.uint8)
        else :
            yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV).astype(np.float32)
            y=yuv[:,:,0]
            y_f=cv2.filter2D(img,ddepth=-1,kernel=kernel,borderType=borderType)
            yuv[:,:,0]=np.clip(y_f,0,255)
            out=cv2.cvtColor(img,cv2.COLOR_YUV2BGR)
            return out

    def exits(self,name):
        return str(name) in self.kernels
    
    def unregister(self,name):
        self.kernels.pop(str(name),None)

    def list(self):
        return sorted(self.kernels.keys())
    
    def __len__(self):
        return len(self.kernels)

# 在模块级创建一个全局的 KernelBank 单例，供整个模块/程序复用
kb = KernelBank()

# 便捷接口：在外部可以调用 register_kernel(...) 注册核/factory
def register(name, kernel=None, factory=None):
    kb.register(name, kernel=kernel, factory=factory)

def set_kernel(name, kernel):
    kb.set_kernel(name, kernel)

def set_factory(name, factory):
    kb.set_factory(name, factory)

def list_kernels():
    return kb.list()

def gaussian_kernel(size=3,sigma=1.0):
    assert size%2==1,"size must be odd"
    ax=np.arange(-size//2+1,size//2+1)
    xx,yy=np.meshgrid(ax,ax)
    kernel=np.exp(-(xx**2+yy**2)/(2*sigma*sigma))
    kernel=kernel/np.sum(kernel)
    return kernel

def average_kernel(size=3):
    """
    生成 size x size 的均值卷积核（归一化）。
    size 必须为奇数，返回 dtype 为 float32 的二维核。
    """
    assert size%2==1,"size must be odd"
    xx=np.ones((size,size),dtype=np.float32)
    xx/=size*size  
    return xx

def sharpen(size=3, sigma=None, amount=1.0):
    """
    生成锐化卷积核（基于 unsharp mask 核形式）：
      kernel = (1+amount)*delta - amount * Gaussian(size, sigma)
    参数:
      size: 奇数核大小
      sigma: 高斯标准差，None 时取 size/6
      amount: 锐化强度（0 不变，越大越锐）
    返回: float32 的 2D 卷积核
    """
    assert size % 2 == 1, "size must be odd"
    if sigma is None:
        sigma = max(0.3, size / 6.0)
    ax = np.arange(-size // 2, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = -amount * g
    kernel[size // 2, size // 2] += (1.0 + amount)
    return kernel.astype(np.float32)

register("gaussian",factory=gaussian_kernel)
register("average",factory=average_kernel)
register("sharpen",factory=sharpen)

def process_image(img,name,*args,kernel=None,factory=None,**kwargs):
    key=str(name)
    if key not in kb.kernels:
        register(name,kernel,factory)

    return kb.apply(img,name,*args,**kwargs)

