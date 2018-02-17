#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import inspect

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_manipulation.data_aug import extend_dataset_flip_axis, dataset_augmentation  # noqa
from data_manipulation.data_aug import binarize_dataset, gray_dataset, green_dataset  # noqa
from data_manipulation.data_mani import data_cut, create_record  # noqa


class DataHolder():
    """
    Class that preprocess all data.
    If the data is already processed, i.e., there is already
    three tfrecords files, you can pass the files as a list
    of paths as the variable "records". Note that in this
    list the expected order is train.tfrecords, valid.tfrecords,
    test.tfrecords.

    :param config: config class with all hyper param information
    :type config: Config
    :param data_path: path to load data np.array
    :type data_path: str
    :param record_path: path to load labels np.array
    :type label_path: str
    :param record_path: path to save tfrecord
    :type record_path: str
    :param flip: param to control if the data
                         will be flipped
    :type flip: boolean
    :param augmentation: param to control if the data
                         will augmented
    :type augmentation: boolean
    :param gray: param to control if the data
                 will be grayscale images
    :type gray: boolean
    :param green: param to control if the data will use only
                  the green channel
    :type green: boolean
    :param binary: param to control if the data will be binarized
    :type binary: boolean
    :param records: list of paths to tfrecords
    :type records: list of str
    """
    def __init__(self,
                 config,
                 data_path=None,
                 label_path=None,
                 record_path=None,
                 flip=False,
                 augmentation=False,
                 gray=False,
                 green=False,
                 binary=False,
                 records=None):
        self.config = config
        self.data_path = data_path
        self.label_path = label_path
        self.record_path = record_path
        self.flip = flip
        self.augmentation = augmentation
        self.gray = gray
        self.green = green
        self.binary = binary
        self.records = records

    def create_records(self):
        """
        Take the arrays in self.data_path and self.label_path,
        divide them into train, test and valid dataset, shufle them,
        and processed them (flip, add augmentation, transform in grayscale,
        transform in image with only the green channel, transform
        in binarized image) and save all records as:
              self.record_path + '_train.tfrecords'
              self.record_path + '_valid.tfrecords'
              self.record_path + '_test.tfrecords'
        """
        if self.gray or self.green or self.binary:
            msg = "only one condition should be True"
            assert self.gray ^ self.green ^ self.binary, msg

        data = np.load(self.data_path)
        labels = np.load(self.label_path)

        # fliping the original data
        # and dividing it into train, test and valid
        if self.flip:
            data, labels = extend_dataset_flip_axis(data,
                                                    labels,
                                                    self.config.height,
                                                    self.config.width,
                                                    self.config.channels)
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels = data_cut(data, labels)  # noqa

        # applying data augmentation to the train dataset
        if self.augmentation:
            train_data, train_labels = dataset_augmentation(train_data,
                                                            train_labels,
                                                            self.config.height,
                                                            self.config.width,
                                                            self.config.channels)  # noqa
        # transforming all images into grayscale
        if self.gray:
            train_data, _ = gray_dataset(train_data,
                                         self.config.height,
                                         self.config.width,
                                         self.config.channels)
            valid_data, _ = gray_dataset(valid_data,
                                         self.config.height,
                                         self.config.width,
                                         self.config.channels)
            test_data, _ = gray_dataset(test_data,
                                        self.config.height,
                                        self.config.width,
                                        self.config.channels)
            self.config.channels = 1

        # transforming all images into images with only green channel
        if self.green:
            train_data, _ = green_dataset(train_data,
                                          self.config.height,
                                          self.config.width,
                                          self.config.channels)
            valid_data, _ = green_dataset(valid_data,
                                          self.config.height,
                                          self.config.width,
                                          self.config.channels)
            test_data, _ = green_dataset(test_data,
                                         self.config.height,
                                         self.config.width,
                                         self.config.channels)
            self.config.channels = 1

        # transforming all images into binary images
        if self.binary:
            train_data, _ = binarize_dataset(train_data,
                                             self.config.height,
                                             self.config.width,
                                             self.config.channels)
            valid_data, _ = binarize_dataset(valid_data,
                                             self.config.height,
                                             self.config.width,
                                             self.config.channels)
            test_data, _ = binarize_dataset(test_data,
                                            self.config.height,
                                            self.config.width,
                                            self.config.channels)
            self.config.channels = 1

        # transforming all data into tf.records
        tfrecords_filename_train = self.record_path + '_train.tfrecords'
        tfrecords_filename_valid = self.record_path + '_valid.tfrecords'
        tfrecords_filename_test = self.record_path + '_test.tfrecords'
        create_record(tfrecords_filename_train,
                      train_data,
                      train_labels,
                      self.config.height,
                      self.config.width,
                      self.config.channels)
        create_record(tfrecords_filename_valid,
                      valid_data,
                      valid_labels,
                      self.config.height,
                      self.config.width,
                      self.config.channels)
        create_record(tfrecords_filename_test,
                      test_data,
                      test_labels,
                      self.config.height,
                      self.config.width,
                      self.config.channels)
        self.records = [tfrecords_filename_train,
                        tfrecords_filename_valid,
                        tfrecords_filename_test]

    def get_train_tfrecord(self):
        """
        retun path to train tf records
        :rtype: str
        """
        if self.records is None:
            return "train.tfrecords"
        else:
            return self.records[0]

    def get_valid_tfrecord(self):
        """
        retun path to valid tf records
        :rtype: str
        """
        if self.records is None:
            return "valid.tfrecords"
        else:
            return self.records[1]

    def get_test_tfrecord(self):
        """
        retun path to test tf records
        :rtype: str
        """
        if self.records is None:
            return "test.tfrecords"
        else:
            return self.records[2]
