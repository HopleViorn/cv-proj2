# CV Project 2: Edge Detection and Image Segmentation

This project implements several common algorithms for edge detection and image segmentation.

## 1. Edge Detection

Navigate to the `EdgeDetection` directory to run the edge detection scripts:
```bash
cd EdgeDetection
```

### Laplacian Operator
To perform edge detection using the Laplacian operator, run:
```bash
python main.py --mode laplacian
```

### Sobel Operator
To perform edge detection using the Sobel operator, run:
```bash
python main.py --mode sobel
```

### Canny Edge Detector
To perform edge detection using the Canny algorithm, run:
```bash
python main.py --mode canny
```

All output images will be saved in the `EdgeDetection/output` directory.

## 2. Image Segmentation

Navigate to the `ImageSegmentation` directory to run the image segmentation scripts:
```bash
cd ImageSegmentation
```

### K-Means Clustering
To perform image segmentation using K-Means clustering, run:
```bash
python custom-k-means.py
```

### Region Growing
To perform image segmentation using the region growing algorithm, run:
```bash
python region-growth.py
```

All output images will be saved in the `ImageSegmentation/output` directory.


