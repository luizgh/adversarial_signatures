from typing import Tuple

import numpy as np
from skimage import filters, transform


def preprocess_signature_otsu_last(img: np.ndarray,
                         canvas_size: Tuple[int, int],
                         img_size: Tuple[int, int] =(170, 242),
                         input_size: Tuple[int, int] =(150, 220)) -> np.ndarray:
    """ Pre-process a signature image, centering it in a canvas, resizing the image,
        cropping it and running otsu to remove the noise.

    Parameters
    ----------
    img : np.ndarray (H x W)
        The signature image
    canvas_size : tuple (H x W)
        The size of a canvas where the signature will be centered on.
        Should be larger than the signature.
    img_size : tuple (H x W)
        The size that will be used to resize (rescale) the signature
    input_size : tuple (H x W)
        The final size of the signature, obtained by croping the center of image.
        This is necessary in cases where data-augmentation is used, and the input
        to the neural network needs to have a slightly smaller size.

    Returns
    -------
    np.narray (input_size):
        The pre-processed image
    -------

    """
    img = img.astype(np.uint8)
    centered, location = normalize_size(img, canvas_size, remove_background=False)
    inverted = 255 - centered
    resized, location = resize_image(inverted, img_size, location)

    noise_removed = remove_background(resized, location)

    if input_size is not None and input_size != img_size:
        cropped = crop_center(noise_removed, input_size)
    else:
        cropped = noise_removed

    return cropped


def normalize_size(img: np.ndarray,
                   canvas_size: Tuple[int, int] = (840, 1360),
                   remove_background: bool = True) -> Tuple[np.ndarray, Tuple]:
    """ Centers an image in a pre-defined canvas size, and remove
    noise using OTSU's method.

    Parameters
    ----------
    img : np.ndarray (H x W)
        The image to be processed
    canvas_size : tuple (H x W)
        The desired canvas size
    remove_background : bool
        Whether to remove the background (with OTSU) or not
    Returns
    -------
    np.ndarray (H x W)
        The normalized image

    tuple (hmin, hmax), (wmin, wmax)
        The location of the signature in the resulting image
    """

    # 1) Crop the image before getting the center of mass

    # Apply a gaussian filter on the image to remove small components
    # Note: this is only used to define the limits to crop the image
    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)

    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    threshold = filters.threshold_otsu(img)

    # Find the center of mass
    binarized_image = blurred_image > threshold
    r, c = np.where(binarized_image == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
    cropped = img[r.min(): r.max(), c.min(): c.max()]

    # 2) Center the image
    img_rows, img_cols = cropped.shape
    max_rows, max_cols = canvas_size

    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center

    # Make sure the new image does not go off bounds
    # Emit a warning if the image needs to be cropped, since we don't want this
    # for most cases (may be ok for feature learning, so we don't raise an error)
    if img_rows > max_rows:
        # Case 1: image larger than required (height):  Crop.
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_rows - max_rows
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_rows, :]
        img_rows = max_rows
    else:
        extra_r = (r_start + img_rows) - max_rows
        # Case 2: centering exactly would require a larger image. relax the centering of the image
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0:
            r_start = 0

    if img_cols > max_cols:
        # Case 3: image larger than required (width). Crop.
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_cols - max_cols
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_cols]
        img_cols = max_cols
    else:
        # Case 4: centering exactly would require a larger image. relax the centering of the image
        extra_c = (c_start + img_cols) - max_cols
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0

    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    # Add the image to the blank canvas
    normalized_image[r_start:r_start + img_rows, c_start:c_start + img_cols] = cropped

    if remove_background:
        # Remove noise - anything higher than the threshold. Note that the image is still grayscale
        normalized_image[normalized_image > threshold] = 255

    location = (r_start, r_start + img_rows), (c_start, c_start + img_cols)
    return normalized_image, location


def crop_center(img: np.ndarray,
                size: Tuple[int, int]) -> np.ndarray:
    """ Crops the center of an image

        Parameters
        ----------
        img : np.ndarray (H x W)
            The image to be cropped
        size: tuple (H x W)
            The desired size

        Returns
        -------
        np.ndarray
            The cRecentropped image
        """
    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = img[start_y: start_y + size[0], start_x:start_x + size[1]]
    return cropped


def remove_background(img: np.ndarray,
                      location) -> np.ndarray:
        """ Remove noise using OTSU's method.

        Parameters
        ----------
        img : np.ndarray
            The image to be processed

        Returns
        -------
        np.ndarray
            The image with background removed
        """

        img = img.astype(np.uint8)
        location_h, location_w = location

        threshold = filters.threshold_otsu(img[location_h[0]:location_h[1],
                                               location_w[0]:location_w[1]])

        # Remove noise - Note that the image is still grayscale
        img[img < threshold] = 0

        return img


def resize_image(img: np.ndarray,
                 size: Tuple[int, int],
                 location) -> Tuple[np.ndarray, Tuple]:
    """ Crops an image to the desired size without stretching it.

    Parameters
    ----------
    img : np.ndarray (H x W)
        The image to be cropped
    size : tuple (H x W)
        The desired size

    Returns
    -------
    np.ndarray
        The cropped image
    tuple (hmin, hmax), (wmin, wmax)
        The location of the signature in the resulting image
    """
    height, width = size

    # Check which dimension needs to be cropped
    # (assuming the new height-width ratio may not match the original size)
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    # Resize the image (will still be larger than new_size in one dimension)
    img = transform.resize(img, (resize_height, resize_width),
                           mode='constant', anti_aliasing=True, preserve_range=True)

    img = img.astype(np.uint8)

    ratio = min(width_ratio, height_ratio)
    location_h = int(location[0][0] / ratio), int(location[0][1] / ratio)
    location_w = int(location[1][0] / ratio), int(location[1][1] / ratio)

    # Crop to exactly the desired new_size, using the middle of the image:
    if width_ratio > height_ratio:
        start = int(round((resize_width-width)/2.0))
        location_w = location_w[0] + start, location_w[1] + start
        return img[:, start:start + width], (location_h, location_w)
    else:
        start = int(round((resize_height-height)/2.0))
        location_h = location_h[0] + start, location_h[1] + start
        return img[start:start + height, :], (location_h, location_w)