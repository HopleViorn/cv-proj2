import cv2
import numpy as np
from ops import numpy_conv2d, numpy_gaussian_blur

def StandSobel(src, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=3):
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)
    
    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel using OpenCV
    sobel_x = cv2.Sobel(src_gray, ddepth, dx, 0, ksize=ksize)
    sobel_y = cv2.Sobel(src_gray, ddepth, 0, dy, ksize=ksize)
    
    # Combine gradients and convert back to uint8
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    return combined

def CustomSobel(src, ddepth=cv2.CV_16S, ksize=3):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    src_gray = numpy_gaussian_blur(src_gray, 3, 1.0)
    
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
    
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0,  0,  0],
                               [1,  2,  1]], dtype=np.float32)
    
    sobel_x = numpy_conv2d(src_gray, sobel_x_kernel)
    sobel_y = numpy_conv2d(src_gray, sobel_y_kernel)
    
    if ddepth == cv2.CV_16S:
        sobel_x = sobel_x.astype(np.int16)
        sobel_y = sobel_y.astype(np.int16)
    
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    return combined