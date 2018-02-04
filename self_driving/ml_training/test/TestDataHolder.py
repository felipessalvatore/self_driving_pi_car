#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import sys
import inspect
import numpy as np
import tensorflow as tf
import itertools

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from util import run_test, reconstruct_from_record  # noqa
from Config import Config  # noqa
from DataHolder import DataHolder  # noqa


class TestDataHolder(unittest.TestCase):
    """
    Class that test the if the data manipulation functions
    """
    @classmethod
    def setUpClass(cls):
        config = Config()
        data_name = "toy_data.npy"
        label_name = "toy_label.npy"
        cls.original_dh = DataHolder(config,
                                     data_name,
                                     label_name,
                                     record_path="toy")
        cls.original_flip = DataHolder(config,
                                       data_name,
                                       label_name,
                                       record_path="toy_flip",
                                       flip=True)
        cls.original_aug = DataHolder(config,
                                      data_name,
                                      label_name,
                                      record_path="toy_aug",
                                      augmentation=True)
        cls.original_gray = DataHolder(config,
                                       data_name,
                                       label_name,
                                       record_path="toy_gray",
                                       gray=True)
        cls.original_green = DataHolder(config,
                                        data_name,
                                        label_name,
                                        record_path="toy_green",
                                        green=True)
        cls.original_binary = DataHolder(config,
                                         data_name,
                                         label_name,
                                         record_path="toy_bin",
                                         binary=True)
        cls.all_dataholders_no_new = [cls.original_dh,
                                      cls.original_gray,
                                      cls.original_green,
                                      cls.original_binary]
        cls.all_paths = ["toy",
                         "toy_flip",
                         "toy_aug",
                         "toy_gray",
                         "toy_green",
                         "toy_bin"]
        cls.original_dh.create_records()
        cls.original_flip.create_records()
        cls.original_aug.create_records()
        cls.original_gray.create_records()
        cls.original_green.create_records()
        cls.original_binary.create_records()

    @classmethod
    def tearDown(cls):
        sufixes = ['_train.tfrecords', '_valid.tfrecords', '_test.tfrecords']
        for car, cdr in itertools.product(cls.all_paths, sufixes):
            file_name = car + cdr
            if os.path.exists(file_name):
                os.remove(file_name)

    def check_size_and_type_data_holder(self,
                                        dataholder,
                                        mode="train",
                                        sizes=[20, 3, 3]):
        if mode == "train":
            size = sizes[0]
            record_path = dataholder.get_train_tfrecord()
        elif mode == "valid":
            size = sizes[1]
            record_path = dataholder.get_valid_tfrecord()
        elif mode == "test":
            size = sizes[2]
            record_path = dataholder.get_test_tfrecord()
        images, labels, shape = reconstruct_from_record(record_path)
        self.assertEqual(images.shape, (size, shape[0] * shape[1] * shape[2]))
        self.assertEqual(labels.shape, (size, 1))
        self.assertEqual(np.uint8, labels.dtype)
        self.assertEqual(np.uint8, images.dtype)

    def test_tf_record_is_created_and_can_be_restored(self):

        modes = ["train", "valid", "test"]

        for dh, mode in itertools.product(self.all_dataholders_no_new, modes):
            self.check_size_and_type_data_holder(dh, mode=mode)
        for mode in modes:
            self.check_size_and_type_data_holder(self.original_flip,
                                                 mode=mode,
                                                 sizes=[25, 4, 4])
            self.check_size_and_type_data_holder(self.original_aug,
                                                 mode=mode,
                                                 sizes=[60, 3, 3])


if __name__ == "__main__":
    run_test(TestDataHolder)
