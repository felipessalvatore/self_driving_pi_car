#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import sys
import inspect
import numpy as np

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from util import run_test, get_image  # noqa
from img2array import create_data_set_as_np_array  # noqa
from data_aug import extend_dataset_flip_axis, binarize_dataset  # noqa
from data_aug import gray_dataset, green_dataset, dataset_augmentation  # noqa


class TestDataAug(unittest.TestCase):
    """
    Class that test the data augmentation functions
    """
    @classmethod
    def setUpClass(cls):
        cls.image_folder = os.path.join(currentdir, "pictures_for_test")
        cls.width = 160
        cls.height = 90
        cls.channels = 3
        cls.data_name = os.path.join(currentdir, "toy_data.npy")
        cls.label_name = os.path.join(currentdir, "toy_label.npy")
        create_data_set_as_np_array(cls.image_folder,
                                    cls.data_name,
                                    cls.label_name,
                                    cls.width,
                                    cls.height,
                                    cls.channels,
                                    verbose=False)
        cls.data = np.load(cls.data_name)
        cls.labels = np.load(cls.label_name)

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.data_name):
            os.remove(cls.data_name)
        if os.path.exists(cls.label_name):
            os.remove(cls.label_name)

    def test_flip_data(self):
        """
        In the toy dataset we have only 7 pictures classified
        as "right". This test checks if the flip function is working
        adding new 7 images with label "left" (2) to the dataset.
        """
        aug_data, aug_labels = extend_dataset_flip_axis(self.data, self.labels)
        data_expected_shape = (25 + 7,
                               self.width * self.height * self.channels)
        self.assertEqual(aug_data.shape, data_expected_shape)
        self.assertEqual(aug_labels.shape, (25 + 7, 1))
        self.assertEqual(np.uint8, aug_labels.dtype)
        self.assertEqual(np.uint8, aug_data.dtype)
        one_right_image = 0
        one_left_image = 25
        original_image = get_image(self.data[one_right_image])
        original_image = np.flip(original_image, axis=1)
        fliped_image = get_image(aug_data[one_left_image])
        condition = np.all(np.equal(original_image, fliped_image))
        msg = "images: {} (orignal) and {} (augmentaded) are not equal".format(one_right_image, one_left_image)  # noqa
        self.assertTrue(condition, msg=msg)
        only_left = aug_labels[25: 25 + 7]
        only_left = only_left.flatten()
        self.assertEqual(np.min(only_left), np.max(only_left))

    def test_one_channel_transformation(self):
        one_channel_shape = (self.data.shape[0],
                             self.width * self.height)
        new_data, shape = binarize_dataset(self.data)
        self.assertEqual(new_data.shape, one_channel_shape)
        self.assertEqual(shape, (self.height, self.width))
        self.assertEqual(np.uint8, new_data.dtype)
        new_data, shape = gray_dataset(self.data)
        self.assertEqual(new_data.shape, one_channel_shape)
        self.assertEqual(shape, (self.height, self.width))
        self.assertEqual(np.uint8, new_data.dtype)
        new_data, shape = green_dataset(self.data)
        self.assertEqual(new_data.shape, one_channel_shape)
        self.assertEqual(shape, (self.height, self.width))
        self.assertEqual(np.uint8, new_data.dtype)

    def test_data_augmentation(self):
        aug_data, aug_labels = dataset_augmentation(self.data, self.labels)
        data_expected_shape = (25 * 3,
                               self.width * self.height * self.channels)
        self.assertEqual(aug_data.shape, data_expected_shape)
        self.assertEqual(aug_labels.shape, (25 * 3, 1))
        self.assertEqual(np.uint8, aug_labels.dtype)
        self.assertEqual(np.uint8, aug_data.dtype)


if __name__ == "__main__":
    run_test(TestDataAug)
