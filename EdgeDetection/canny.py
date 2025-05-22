import cv2
import numpy as np

def StandCanny(img, lowThreshold, highThreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, lowThreshold, highThreshold)
    return edges

def CustomCanny(blurred, lowThreshold, highThreshold):
    # Your Job
    return edges