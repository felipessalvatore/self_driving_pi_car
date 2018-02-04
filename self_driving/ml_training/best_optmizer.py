#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def optmizers_search(channels,
                     records=None):
    """
    Script to run optmizers search,
    the result is saved on the file optmizers_results.txt

    :param channels: image channels
    :type channels: int
    :param records: list of paths to tf_records
    :type records: none or list of str
    """
    OT = [tf.train.GradientDescentOptimizer,
          tf.train.AdadeltaOptimizer,
          tf.train.AdagradOptimizer,
          tf.train.AdamOptimizer,
          tf.train.FtrlOptimizer,
          tf.train.ProximalGradientDescentOptimizer,
          tf.train.ProximalAdagradOptimizer,
          tf.train.RMSPropOptimizer]

    OT_name = ["GradientDescentOptimizer",
               "AdadeltaOptimizer",
               "AdagradOptimizer",
               "AdamOptimizer",
               "FtrlOptimizer",
               "ProximalGradientDescentOptimizer",
               "ProximalAdagradOptimizer",
               "RMSPropOptimizer"]
    numeric_result = []
    results = []
    info = []

    for name, opt in zip(OT_name, OT):
        config = Config(channels=channels,
                        optimizer=opt)
        data = DataHolder(config,
                          records=records)
        print(name + ":\n")
        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph, config, network, data)
        trainer.fit(verbose=True)
        valid_acc = trainer.get_valid_accuracy()
        numeric_result.append(valid_acc)
        name += ': valid_acc = {0:.6f} | '.format(valid_acc)
        test_images, test_labels, _ = reconstruct_from_record(data.get_test_tfrecord())  # noqa
        test_images = test_images.astype(np.float32) / 255
        test_pred = trainer.predict(test_images)
        acc_cat = accuracy_per_category(test_pred, test_labels, categories=4)
        for i, cat_result in enumerate(acc_cat):
            name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
        results.append(name)
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        attrs = vars(config)
        config_info = ["%s: %s" % item for item in attrs.items()]
        info.append(config_info)

    best_result = max(list(zip(numeric_result, OT_name, info)))
    result_string = """In an experiment with different optmizers
    the best one is {0} with valid accuracy of {1}.
    \nThe training uses the following params:
    \n{2}\n""".format(best_result[1],
                      best_result[0],
                      best_result[2])
    file = open("optmizers_results.txt", "w")
    file.write("Results for different optmizers\n")
    for result in results:
        result += "\n"
        file.write(result)
    file.write("\n")
    file.write(result_string)
    file.close()


def main():
    """
    Main script to perform learnig rate search.

    "mode" is the argument to choose which kind of data will be used:
        "pure": rgb image with no manipulation.
        "flip": flippped rgb image (a image with label "left" is
                flipped and transform in an image with label
                "right", and vice versa; to have a balanced data).
        "aug": flippped rgb image with new shadowed and blurred images.
        "bin": binary image, only one channel.
        "gray": grayscale image, only one channel.
        "green": image with only the green channel.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="pure",
                        help="mode for data: pure, flip, aug, bin, gray, green (default=pure)")  # noqa
    args = parser.parse_args()
    if args.mode == "bin" or args.mode == "gray" or args.mode == "green":
        channels = 1
    else:
        channels = 3
    records = ["_train.tfrecords", "_valid.tfrecords", "_test.tfrecords"]
    new_records = []
    for record in records:
        record = args.mode + record
        new_records.append(record)
    optmizers_search(channels=channels,
                     records=new_records)


if __name__ == "__main__":
    main()
