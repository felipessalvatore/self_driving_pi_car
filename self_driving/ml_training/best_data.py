'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
import tensorflow as tf
import os
import numpy as np
import shutil
import argparse

from DataHolder import DataHolder
from Config import Config
from Trainer import Trainer
from DFN import DFN
from util import reconstruct_from_record, accuracy_per_category
from util import int2command


def data_search(data_path=None,
                label_path=None,
                have_records=False):
    """
    Script to search different types of data, here we use
    the simplest --model sofmax classifier--,
    the result is saved on the file data_results.txt

    :param data_path: path to data as numpy array
    :type data_path: str
    :param label_path: path to labels as numpy array
    :type label_path: str
    :param have_records: param to control if tf_records will be created
                         or not.
    :type have_records: bool
    """
    config_pure = Config(architecture=[4])
    config_flip = Config(architecture=[4])
    config_aug = Config(architecture=[4])
    config_bin = Config(architecture=[4], channels=1)
    config_green = Config(architecture=[4], channels=1)
    config_gray = Config(architecture=[4], channels=1)
    data_pure = DataHolder(config_pure,
                           data_path=data_path,
                           label_path=label_path,
                           record_path="pure",
                           records=["pure_train.tfrecords",
                                    "pure_valid.tfrecords",
                                    "pure_test.tfrecords"])
    data_flip = DataHolder(config_flip,
                           data_path=data_path,
                           label_path=label_path,
                           record_path="flip",
                           flip=True,
                           records=["flip_train.tfrecords",
                                    "flip_valid.tfrecords",
                                    "flip_test.tfrecords"])
    data_aug = DataHolder(config_aug,
                          data_path=data_path,
                          label_path=label_path,
                          record_path="aug",
                          flip=True,
                          augmentation=True,
                          records=["aug_train.tfrecords",
                                   "aug_valid.tfrecords",
                                   "aug_test.tfrecords"])

    data_bin = DataHolder(config_bin,
                          data_path=data_path,
                          label_path=label_path,
                          record_path="bin",
                          flip=True,
                          binary=True,
                          records=["bin_train.tfrecords",
                                   "bin_valid.tfrecords",
                                   "bin_test.tfrecords"])
    data_green = DataHolder(config_green,
                            data_path=data_path,
                            label_path=label_path,
                            record_path="green",
                            flip=True,
                            green=True,
                            records=["green_train.tfrecords",
                                     "green_valid.tfrecords",
                                     "green_test.tfrecords"])
    data_gray = DataHolder(config_gray,
                           data_path=data_path,
                           label_path=label_path,
                           record_path="gray",
                           flip=True,
                           green=True,
                           records=["gray_train.tfrecords",
                                    "gray_valid.tfrecords",
                                    "gray_test.tfrecords"])
    all_data = [data_pure,
                data_flip,
                data_aug,
                data_bin,
                data_green,
                data_gray]

    all_config = [config_pure,
                  config_flip,
                  config_aug,
                  config_bin,
                  config_green,
                  config_gray]

    names = ["data with no augmentation",
             "fliped augmentation",
             "data with augmentation",
             "binarized data",
             "data with only green channel",
             "grayscale data"]
    results = []
    for data, config, name in zip(all_data, all_config, names):
        print(name + ":\n")
        if not have_records:
            assert data_path is not None
            assert label_path is not None
            data.create_records()
        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph, config, network, data)
        trainer.fit(verbose=True)
        valid_acc = trainer.get_valid_accuracy()
        name += ': valid_acc = {0:.6f} | '.format(valid_acc)
        valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord()) # noqa
        valid_images = valid_images.astype(np.float32) / 255
        valid_pred = trainer.predict(valid_images)
        acc_cat = accuracy_per_category(valid_pred, valid_labels, categories=4)
        for i, cat_result in enumerate(acc_cat):
            name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
        results.append(name)
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")

    file = open("data_results.txt", "w")
    file.write("Results with different data\n")
    for result in results:
        result += "\n"
        file.write(result)
    file.close()


def main():
    """
    Main script to perform data search.
    """
    parser = argparse.ArgumentParser(description='Perform data search')
    parser.add_argument("-d",
                        "--train_data",
                        type=str,
                        default=None,
                        help="train data path (default=None)")
    parser.add_argument("-l",
                        "--train_label",
                        type=str,
                        default=None,
                        help="label data path (default=None)")
    args = parser.parse_args()
    cond1 = type(args.train_data) == str
    cond2 = type(args.train_label) == str
    have_records = not (cond1 and cond2)
    data_search(args.train_data,
                args.train_label,
                have_records)


if __name__ == "__main__":
    main()
