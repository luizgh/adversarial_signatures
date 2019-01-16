import cv2
import numpy as np


def remove_small_components(img, min_component_size, background_color=0):
    n_components, output, stats, centroids = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    new_img = np.ones_like(img) * background_color

    for idx in range(1, len(stats)): # idx 0 is background; start with 1
        if stats[idx, -1] > min_component_size: # last column of stats contain the area (# pixels)
            new_img[output == idx] = img[output == idx]
    return new_img


def run_otsu(img):
    new_img = img.copy()
    new_img_quantized = new_img.astype(np.uint8)
    threshold, _ = cv2.threshold(new_img_quantized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_img[new_img_quantized <= threshold] = 0
    return new_img
