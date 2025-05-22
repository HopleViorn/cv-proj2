import cv2
import numpy as np
from ops import numpy_gaussian_blur, numpy_conv2d

def StandCanny(img, lowThreshold, highThreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, lowThreshold, highThreshold)
    return edges

def CustomCanny(img, lowThreshold, highThreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = numpy_gaussian_blur(gray, kernel_size=5, sigma=1.4)
    
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float64)
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float64)

    grad_x = numpy_conv2d(blurred, sobel_x_kernel)
    grad_y = numpy_conv2d(blurred, sobel_y_kernel)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle = np.abs(angle) % 180
    
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                neighbors = [magnitude[i,j-1], magnitude[i,j+1]]
            elif (22.5 <= angle[i,j] < 67.5):
                neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
            elif (67.5 <= angle[i,j] < 112.5):
                neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
            else:
                neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
            
            if magnitude[i,j] >= max(neighbors):
                suppressed[i,j] = magnitude[i,j]
    
    strong_edges = (suppressed >= highThreshold)
    weak_edges = (suppressed >= lowThreshold) & (suppressed < highThreshold)
    
    edges = np.zeros_like(suppressed, dtype=np.uint8)
    edges[strong_edges] = 255
    
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if weak_edges[i,j]:
                if (edges[i-1:i+2, j-1:j+2] > 0).any():
                    edges[i,j] = 255
    
    return edges