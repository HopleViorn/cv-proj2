import cv2
import numpy as np

def StandCanny(img, lowThreshold, highThreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, lowThreshold, highThreshold)
    return edges

def CustomCanny(img, lowThreshold, highThreshold):
    # 1. 转换为灰度图并应用高斯滤波
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # 2. 使用Sobel算子计算梯度幅值和方向
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle = np.abs(angle) % 180
    
    # 3. 非极大值抑制
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            # 根据梯度方向确定相邻像素
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                neighbors = [magnitude[i,j-1], magnitude[i,j+1]]
            elif (22.5 <= angle[i,j] < 67.5):
                neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
            elif (67.5 <= angle[i,j] < 112.5):
                neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
            else:
                neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
            
            # 如果当前像素是局部最大值则保留
            if magnitude[i,j] >= max(neighbors):
                suppressed[i,j] = magnitude[i,j]
    
    # 4. 双阈值检测
    strong_edges = (suppressed >= highThreshold)
    weak_edges = (suppressed >= lowThreshold) & (suppressed < highThreshold)
    
    # 5. 边缘连接分析
    edges = np.zeros_like(suppressed, dtype=np.uint8)
    edges[strong_edges] = 255
    
    # 8邻域连接分析
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if weak_edges[i,j]:
                if (edges[i-1:i+2, j-1:j+2] > 0).any():
                    edges[i,j] = 255
    
    return edges