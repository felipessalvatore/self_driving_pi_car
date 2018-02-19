#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import sys
import inspect
try:
    from util import get_image_and_command, get_image, get_flat_shape
    from util import load_dataset, save_dataset
except ImportError:
    from data_manipulation.util import get_image_and_command, get_image, get_flat_shape  # noqa
    from data_manipulation.util import load_dataset, save_dataset

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import vision.image_manipulation as img_mani  # noqa


def extend_dataset_flip_axis(data,
                             labels,
                             height=90,
                             width=160,
                             channels=3):
    """
    Balance and extend dataset
    by generating new images flipping the horizontal
    axis (only applicable to images labeled 'left' or 'right').
    This function is hard-coded, it assumes the following codification:
        - "up": 0
        - "left": 1
        - "right": 2

    :param data: dataset
    :type data: np.array
    :param label: labels
    :type label: np.array
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :return: extended images, extended labels
    :rtype: np.array, np.array
    """
    all_images = []
    all_labels = []
    flat_shape = data.shape[1]
    for i in range(data.shape[0]):
        orig_label = labels[i]
        if orig_label == 0:
            continue
        frame, cmd = get_image_and_command(data[i],
                                           labels[i],
                                           height,
                                           width,
                                           channels)
        if orig_label == 1:
            flip_cmd = 2
        else:
            flip_cmd = 1
        flip = np.flip(frame, axis=1)
        flip = np.array(flip.reshape(flat_shape))
        all_images.append(flip)
        all_labels.append(flip_cmd)
    all_labels = np.array(all_labels).astype('uint8')
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    extended_images = np.concatenate((data, all_images), axis=0)
    extended_labels = np.concatenate((labels, all_labels), axis=0)
    return extended_images, extended_labels


def transfor_dataset_with_one_channel(data,
                                      transformation,
                                      height=90,
                                      width=160,
                                      channels=3):
    """
    Create a new dataset by applying a function "transformation"
    available at vision.image_manipulation.
    Returns a new dataset and the new shape of its contents.
    The new shape will have only height and width.

    :param transformation: function
    :type transformation: np.array -> np.array
    :param data: dataset
    :type data: np.array
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :return: transformed dataset, shape
    :rtype: np.array, tuple
    """
    new_dataset = []
    new_shape = ()
    for i in range(data.shape[0]):
        image = get_image(data[i],
                          height,
                          width,
                          channels)
        new_image = transformation(image)
        if new_shape == ():
            new_shape = new_image.shape
        new_image = new_image.reshape(get_flat_shape(new_image))
        new_dataset.append(new_image)
    new_dataset = np.array(new_dataset).astype('uint8')
    return new_dataset, new_shape


def binarize_dataset(data,
                     height=90,
                     width=160,
                     channels=3):
    """
    Create a new dataset by applying the function binarize_image.

    :param data: dataset
    :type data: np.array
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :return: transformed dataset, shape
    :rtype: np.array, tuple
    """
    data, shape = transfor_dataset_with_one_channel(data,
                                                    img_mani.binarize_image,
                                                    height,
                                                    width,
                                                    channels)
    return data, shape


def gray_dataset(data,
                 height=90,
                 width=160,
                 channels=3):
    """
    Create a new dataset by applying the function grayscale_image.

    :param data: dataset
    :type data: np.array
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :return: transformed dataset, shape
    :rtype: np.array, tuple
    """
    data, shape = transfor_dataset_with_one_channel(data,
                                                    img_mani.grayscale_image,
                                                    height,
                                                    width,
                                                    channels)
    return data, shape


def green_dataset(data,
                  height=90,
                  width=160,
                  channels=3):
    """
    Create a new dataset by applying the function green_channel.

    :param data: dataset
    :type data: np.array
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :return: transformed dataset, shape
    :rtype: np.array, tuple
    """
    data, shape = transfor_dataset_with_one_channel(data,
                                                    img_mani.green_channel,
                                                    height,
                                                    width,
                                                    channels)
    return data, shape


def dataset_augmentation(data, labels, height=90, width=160, channels=3):
    """
    Augment a dataset by inserting a vertical random shadow and
    by bluring with a Gaussian convolution

    :param data: dataset
    :type data: np.array
    :param labels: labels
    :type labels: np.array
    :param width: image width
    :type width: int
    :param height: image height
    :type heights: int
    :param channels: image channels
    :type channels: int
    :return: extended images, extended labels
    :rtype: np.array, np.array
    """
    all_images = []
    all_labels = []
    size = data.shape[0]
    flat_shape = data.shape[1]
    for i in range(size):
        image = get_image(data[i], height, width, channels)
        new_image = img_mani.random_shadow(image)
        new_image = new_image.reshape(flat_shape)
        new_label = labels[i]
        all_images.append(new_image)
        all_labels.append(new_label)
        new_image = img_mani.gaussian_blur(image)
        new_image = new_image.reshape(flat_shape)
        all_images.append(new_image)
        all_labels.append(new_label)
    all_labels = np.array(all_labels).astype('uint8')
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    extended_images = np.concatenate((data, all_images), axis=0)
    extended_labels = np.concatenate((labels, all_labels), axis=0)
    return extended_images, extended_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str, help='path to current data')
    parser.add_argument('labels_path',
                        type=str, help='path to current labels')
    parser.add_argument('new_data_folder_path',
                        type=str, help='path to data and labels to be saved')  # noqa
    parser.add_argument('dataset_name',
                        nargs='?', default='dataset', type=str, help='name for dataset. (Default) dataset')  # noqa

    user_args = parser.parse_args()

    data, labels = load_dataset(user_args.data_path,
                                user_args.labels_path)
    data, labels = extend_dataset_flip_axis(data,
                                            labels)
    data, labels = dataset_augmentation(data, labels)
    data_shape = (90, 160, 3)
    save_dataset(data,
                 labels,
                 user_args.new_data_folder_path,
                 data_shape,
                 user_args.dataset_name)


if __name__ == '__main__':
    main()
