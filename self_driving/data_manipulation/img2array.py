#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import os
import cv2
import numpy as np
from util import command2int, save_dataset


def folder2array(folder_path,
                 pickle_path,
                 height,
                 width,
                 channels,
                 resize,
                 verbose):
    """
    Function to transform all images from the folder folder_name
    into a tuple of np arrays.

    :param folder_path: path to folder containing images
    :type folder_path: str
    :param pickle_path: path to pickle containing the labels
    :type pickle_path: str
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int    
    :param resize: percentage to scale the image
    :type resize: int

    :rtype: (np.array,np.array,np.array)
    """
    all_images = []
    all_labels = []
    flat_shape = height * width * channels
    shape = (height, width, channels)
    if resize != 100: 
        resized_shape = (int((height * resize)/100.0), int((width * resize)/100.0), channels)
        flat_shape = new_shape[0] * new_shape[1] * new_shape[2]
        shape = resized_shape
    resize = resize / 100.0
    with open(pickle_path, "rb") as f:
        label_dict = pickle.load(f)
    if verbose:
        print("Trying to convert images from {} \n".format(folder_path))
    for filename in os.listdir(folder_path):
        key = filename[:- 4]
        label = command2int[label_dict[key]]
        image_path = os.path.join(folder_path, filename)
        image = change_type_to_uint8(cv2.imread(image_path))
        image = cv2.resize(image, (0,0), fx=resize, fy=resize)
        image = image.reshape(flat_shape)
        all_images.append(image)
        all_labels.append(label)
    all_labels = change_type_to_uint8(np.array(all_labels))
    all_images = np.array(all_images)
    return all_images, all_labels, shape


def change_type_to_uint8(image):
    """
    Change type to uint8 Unsigned integer (0 to 255)

    :param image: image as an np.array
    :type image: np.array
    :rtype: np.array
    """
    image = image.astype('uint8')
    return image


def create_data_set_as_np_array(folder_path,
                                npy_path,
                                npy_name="data",
                                height=90,
                                width=160,
                                channels=3,
                                resize=100,
                                verbose=True):
    """
    Giving one path to a folder of folders of images,
    this function transform all images in two arrays
    one with all the flatted images 'npy_name'_<np.shape>_data.npy
    and other with all the respective labels 'npy_name'_<np.shape>_labels.npy
    both saved in 'npy_path'. 

    :param folder_path: path to folder containing folders of images
                        and pickles
    :type folder_path: str
    :param npy_path: name of the data and labels array to be saved
    :type npy_path: str
    :param npy_name: path to data and labels array to be saved
    :type npy_name: str
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param resize: percentage to scale the image
    :type resize: int
    :param verbose: param to print path information
    :type verbose: boolean
    """
    assert os.path.exists(folder_path)
    all_images = []
    all_labels = []
    for folder in os.listdir(folder_path):
        folder = os.path.join(folder_path, folder)
        if os.path.isdir(folder):
            pickle_path = folder + "_pickle"
            images, labels, shape = folder2array(folder,
                                          pickle_path,
                                          height,
                                          width,
                                          channels,
                                          resize,
                                          verbose)
            all_images.append(images)
            all_labels.append(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    save_dataset(all_images, all_labels, npy_path, shape, npy_name)


def main():
    """
    Script to transform one folder containing folders of images
    and pickles to a tuple of np.arrays
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('img_folder_path',
                        type=str, help='path to image folder')
    parser.add_argument('npy_folder_path',
                        type=str, help='path to npy files to be saved')
    parser.add_argument('npy_name',
                        type=str, default="data", help='name of npy files (dafault="data")')
    parser.add_argument("-H",
                        "--image_height",
                        type=int,
                        default=90,
                        help="height number (default=90)")
    parser.add_argument("-w",
                        "--image_width",
                        type=int,
                        default=160,
                        help="width number (default=160)")
    parser.add_argument("-c",
                        "--image_channels",
                        type=int,
                        default=3,
                        help="number of channels (default=3)")
    parser.add_argument("-r",
                        "--resize",
                        type=int,
                        default=100,
                        help="percentage to resize images in dataset (default=100)")
    user_args = parser.parse_args()
    create_data_set_as_np_array(user_args.img_folder_path,
                                user_args.npy_folder_path,
                                user_args.npy_name,
                                user_args.image_height,
                                user_args.image_width,
                                user_args.image_channels,
                                user_args.resize)

if __name__ == '__main__':
    main()
