'''
Useful functions for data augmentation of images
'''

import cv2
import numpy as np
from PIL import Image


def grayscale_image(input_image):
    """
    Convert input_image to grayscale

    :type input_image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)


def binarize_image(input_image, threshold_value=177):
    """
    Convert input_image to binary representation

    :type input_image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    gray_image = grayscale_image(input_image)
    bin_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, bin_image = cv2.threshold(bin_image,
                                 threshold_value,
                                 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_image


def green_channel(input_image):
    """
    Split input_image channels and return only the green channel

    :type input_image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    return input_image[:, :, 1]


def top_bottom_cut(input_image):
    """
    Cut off randomly part
    of the top and bottom of
    input_image and reshape it to the original dimensions

    :type input_image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    height = input_image.shape[0]
    width = input_image.shape[1]
    input_dtype = input_image.dtype
    top = int(np.random.uniform(.325, .425) * height)
    bottom = int(np.random.uniform(.075, .175) * height)
    input_image = input_image[top:-bottom, :]
    img = Image.fromarray(input_image)
    img = img.resize((width, height), Image.LANCZOS)
    cut_image = np.array(img).astype(input_dtype)
    return cut_image


def random_shadow(input_image):
    """
    Insert a vertical random shadow in an input_image

    :type input_image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    height, width = input_image.shape[0], input_image.shape[1]
    [x1, x2] = np.random.choice(width, size=2, replace=False)
    k = height / float(x2 - x1)
    b = - k * x1
    im_array = input_image.copy()
    for i in range(height):
        c = int((i - b) / k)
        im_array[i, :c, :] = (im_array[i, :c, :] * .5).astype(np.uint8)
    return im_array


def gaussian_blur(input_image,
                  kernel_size=5):
    """
    Blur input_image with a Gaussian convolution

    :type input_image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    blur = cv2.GaussianBlur(input_image, (kernel_size, kernel_size), 0)
    return blur
