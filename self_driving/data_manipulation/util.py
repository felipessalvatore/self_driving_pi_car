#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import os

command2int = {"up": 0, "left": 1, "right": 2}
int2command = {i[1]: i[0] for i in command2int.items()}


def run_test(testClass):
    """
    Function to run all the tests from a class of tests.

    :param testClass: class for testing
    :type testClass: unittest.TesCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def get_image_and_command(data_index,
                          label_index,
                          height=90,
                          width=160,
                          channels=3):
    """
    Get and reshape image
    with parameters: 90(height), 160(width), 3(channels)
    from data_index and get it's label
    in label_index (e.g. 'right')

    :type data_index: numpy.ndarray
    :type label_index: numpy.ndarray
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :rtype: numpy.ndarray, str
    """
    img_array = data_index.reshape((height, width, channels))
    command = int2command[label_index[0]]
    return img_array, command


def get_image(data_index,
              height=90,
              width=160,
              channels=3):
    """
    Get and reshape image with parameters:
    90(height), 160(width), 3(channels)
    from data_index

    :type data_index: numpy.ndarray
    :rtype: numpy.ndarray
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    """
    return data_index.reshape((height, width, channels))


def get_flat_shape(image):
    """
    Multiply each shape
    component of image (tuple of array dimensions)

    :type image: numpy.ndarray
    :rtype: int
    """
    flat = 1
    for i in range(len(image.shape)):
        flat *= image.shape[i]
    return flat


def shape2filename(data_shape):
    """
    Get each shape component and return a string
    formatted to 'height_width_channels_'

    :type data_shape: numpy.ndarray
    :rtype: str
    """
    name = ""
    for i in data_shape:
        name += "{}_".format(i)
    return name


def load_dataset(data_path,
                 labels_path):
    """
    Load and return dataset
    arrays from data_path and
    label arrays from labels_path

    :type data_path: str (.npy file)
    :type labels_path: str (.npy file)
    :rtype: numpy.ndarray, numpy.ndarray
    """
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels


def save_dataset(data,
                 labels,
                 folder_path,
                 data_shape,
                 name):
    """
    Save data and labels in a directory as a numpy array binary file (NPY)

    :type data: numpy.ndarray
    :type labels: numpy.ndarray
    :type folder_path: str
    :type data_shape: tuple of numpy array dimensions
    :type name: str
    :rtype:
    """
    shape = shape2filename(data_shape)
    data_name = "{}_{}data.npy".format(name, shape)
    label_name = "{}_{}labels.npy".format(name, shape)
    data_path = os.path.join(folder_path, data_name)
    labels_path = os.path.join(folder_path, label_name)
    np.save(data_path, data)
    np.save(labels_path, labels)
