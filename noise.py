import numpy as np
from skimage import filters


def run_otsu(img):
    new_img = img.copy()
    new_img_quantized = new_img.astype(np.uint8)

    threshold = filters.threshold_otsu(img)

    new_img[new_img_quantized <= threshold] = 0
    return new_img
