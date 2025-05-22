import cv2
import numpy as np

# https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
def StandLaplacian(src, ddepth = cv2.CV_16S, kernel_size=3):
    # laplacian = cv2.Laplacian(image, cv2.CV_8U)

    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Apply Laplace function
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)

    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)

    return abs_dst

def CustomLaplacian(src, ddepth = cv2.CV_16S, kernel_size=3):
    # Your Job
    # Remove noise by blurring with a Gaussian filter
    src_blur = cv2.GaussianBlur(src, (3, 3), 0)

    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src_blur, cv2.COLOR_BGR2GRAY)

    standard_laplacian_3x3 = np.array([[0,  1, 0],
                                       [1, -4, 1],
                                       [0,  1, 0]], dtype=np.float32)

    if kernel_size == 3:
        kernel = standard_laplacian_3x3
    else:
        kernel = standard_laplacian_3x3

    dst_custom = cv2.filter2D(src_gray, ddepth, kernel)

    abs_dst_custom = cv2.convertScaleAbs(dst_custom)

    return abs_dst_custom