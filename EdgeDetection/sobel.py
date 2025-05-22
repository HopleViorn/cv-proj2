import cv2
import numpy as np

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
    # Remove noise by blurring with a Gaussian filter
    src_blur = cv2.GaussianBlur(src, (3, 3), 0)
    
    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src_blur, cv2.COLOR_BGR2GRAY)
    
    # Define Sobel kernels
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
    
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0,  0,  0],
                               [1,  2,  1]], dtype=np.float32)
    
    # Apply custom Sobel filters
    sobel_x = cv2.filter2D(src_gray, ddepth, sobel_x_kernel)
    sobel_y = cv2.filter2D(src_gray, ddepth, sobel_y_kernel)
    
    # Calculate gradient magnitude
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    return combined