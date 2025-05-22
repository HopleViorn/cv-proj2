import numpy as np

def gaussian_kernel(size=3, sigma=1.0):
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return kernel / kernel.sum()

def numpy_gaussian_blur(image, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    return numpy_conv2d(image, kernel)

def numpy_conv2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                   mode='reflect')
    
    shape = (image.shape[0], image.shape[1], kh, kw)
    strides = padded.strides + padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides)
    
    output = np.einsum('ijkl,kl->ij', windows, kernel)
    
    return output.astype(np.float32)