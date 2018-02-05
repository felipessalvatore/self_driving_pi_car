#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import sys
import inspect
import numpy as np
import tensorflow as tf
import itertools
import shutil

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from util import run_test, reconstruct_from_record  # noqa
from Config import Config  # noqa
from DataHolder import DataHolder  # noqa
from DFN import DFN  # noqa
from Trainer import Trainer  # noqa


class TestTrainer(unittest.TestCase):
    """
    Class that test the Trainer class in optimization and prediction
    """
    @classmethod
    def setUpClass(cls):
        data_name = "toy_data.npy"
        label_name = "toy_label.npy"

        cls.config3d = Config(epochs=1,
                              architecture=[4],
                              num_steps=100,
                              save_step=10)
        cls.config1d = Config(epochs=1,
                              architecture=[4],
                              num_steps=100,
                              save_step=10)
        cls.data_aug = DataHolder(cls.config3d,
                                  data_name,
                                  label_name,
                                  record_path="toy_aug",
                                  flip=True,
                                  augmentation=True)
        cls.data_gray = DataHolder(cls.config1d,
                                   data_name,
                                   label_name,
                                   record_path="toy_gray",
                                   flip=True,
                                   augmentation=False,
                                   gray=True)
        cls.data_green = DataHolder(cls.config1d,
                                    data_name,
                                    label_name,
                                    record_path="toy_green",
                                    flip=True,
                                    augmentation=False,
                                    green=True)
        cls.data_binary = DataHolder(cls.config1d,
                                     data_name,
                                     label_name,
                                     flip=True,
                                     augmentation=False,
                                     record_path="toy_bin",
                                     binary=True)
        cls.data_aug.create_records()
        cls.data_gray.create_records()
        cls.data_green.create_records()
        cls.data_binary.create_records()
        cls.all_paths = ["toy_aug",
                         "toy_gray",
                         "toy_green",
                         "toy_bin"]
        cls.data_list = [cls.data_gray, cls.data_green, cls.data_binary]
        cls.end = False

    @classmethod
    def tearDown(cls):
        if cls.end:
            sufixes = ['_train.tfrecords', '_valid.tfrecords', '_test.tfrecords']  # noqa
            for car, cdr in itertools.product(cls.all_paths, sufixes):
                file_name = car + cdr
                if os.path.exists(file_name):
                    os.remove(file_name)

        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")

    def check_overfitting_valid_data(self,
                                     config,
                                     dataholder):
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")

        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph,
                          config,
                          network,
                          dataholder)
        non_trained_acc = trainer.get_valid_accuracy()
        trainer.fit(verbose=False)
        trained_acc = trainer.get_valid_accuracy()
        condition = non_trained_acc < trained_acc
        msg = "Performance on valid data not better after training\n"
        msg += " non_trained_acc = {0:.6f}".format(non_trained_acc)
        msg += " | trained_acc = {0:.6f}".format(trained_acc)
        self.assertTrue(condition, msg=msg)

    def check_prediction(self, config, dataholder, num_classes=4):
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        record_path = dataholder.get_test_tfrecord()
        images, _, shape = reconstruct_from_record(record_path)
        images = images.astype(np.float32) / 255
        num_images = images.shape[0]
        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph,
                          config,
                          network,
                          dataholder)
        non_trained_predictions = trainer.predict(images)
        trainer.fit(verbose=False)
        trained_predictions = trainer.predict(images)
        image = images[0].reshape((1, images[0].shape[0]))
        single_prediction = trainer.predict(image)
        self.assertEqual(non_trained_predictions.shape, (num_images,))
        self.assertEqual(trained_predictions.shape, (num_images,))
        self.assertEqual(np.int32, non_trained_predictions.dtype)
        self.assertEqual(np.int32, trained_predictions.dtype)
        self.assertEqual(np.int32, single_prediction.dtype)

    def test_model_is_fitting_valid_dataset(self):
        self.check_overfitting_valid_data(self.config3d,
                                          self.data_aug)
        for dh in self.data_list:
            self.check_overfitting_valid_data(self.config1d,
                                              dh)

    def test_prediction(self):
        self.check_prediction(self.config3d,
                              self.data_aug)
        for dh in self.data_list:
            self.check_prediction(self.config1d,
                                  dh)
        TestTrainer.end = True  # hack to use TearDown only here


if __name__ == "__main__":
    run_test(TestTrainer)
