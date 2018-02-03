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
from util import int2command, get_random_architecture_and_activations


def architecture_search(records,
                        channels,
                        experiments,
                        deepest_net_size):
    """
    Script to search different architectures for a DFN,
    the result is saved on the file architecture_results.txt

    :param records: list of paths to train, valid and test tf.records
    :type records: list of str
    :param channels: image channels
    :type channels: int
    :param experiments: number of experiments to be made
    :type experiments: int
    :param deepest_net_size: size of the deepest network
    :type deepest_net_size: int
    """
    sizes = np.random.randint(1, deepest_net_size, experiments)
    hidden_layers, activations = get_random_architecture_and_activations(network_sizes=sizes)  # noqa
    numeric_result = []
    results = []
    info = []

    for arch, act in zip(hidden_layers, activations):
        config = Config(architecture=arch,
                        activations=act,
                        channels=channels)

        data = DataHolder(config,
                          records=records)
        name = str(arch)
        print(name + ":\n")
        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph, config, network, data)
        trainer.fit(verbose=True)
        valid_acc = trainer.get_valid_accuracy()
        numeric_result.append(valid_acc)
        name += ': valid_acc = {0:.6f} | '.format(valid_acc)
        valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord())  # noqa
        valid_images = valid_images.astype(np.float32) / 255
        valid_pred = trainer.predict(valid_images)
        acc_cat = accuracy_per_category(valid_pred, valid_labels, categories=4)
        for i, cat_result in enumerate(acc_cat):
            name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
        results.append(name)
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        attrs = vars(config)
        config_info = ["%s: %s" % item for item in attrs.items()]
        info.append(config_info)

    best_result = max(list(zip(numeric_result, hidden_layers, info)))
    result_string = """In an experiment with {0} architectures
    the best one is {1} with valid accuracy of {2}.
    \nThe training uses the following params:
    \n{3}\n""".format(experiments,
                      best_result[1],
                      best_result[0],
                      best_result[2])
    file = open("architecture_results.txt", "w")
    file.write("Results for different architectures\n")
    for result in results:
        result += "\n"
        file.write(result)
    file.write("\n")
    file.write(result_string)
    file.close()


def main():
    """
    Main script to perform architecture search.

    "mode" is the argument to choose which kind of data will be used:
        "pure": rgb image with no manipulation.
        "flip": flippped rgb image (a image with label "left" is
                flipped and transform in an image with label
                "right", and vice versa; to have a balanced data).
        "aug": flippped rgb image with new shadowed and blurred images.
        "bin": binary image, only one channel.
        "gray": grayscale image, only one channel.
        "green": image with only the green channel.

    "experiment" is the number of experiments to be done.

    "deep" is the deep of the model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="pure",
                        help="mode for data: pure, flip, aug, bin, gray, green (default=pure)")  # noqa
    parser.add_argument("-e",
                        "--experiments",
                        type=int,
                        default=10,
                        help="number of experiments to be done (default=10)")  # noqa
    parser.add_argument("-d",
                        "--deep",
                        type=int,
                        default=4,
                        help="deep of the model (default=4)")  # noqa
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
    architecture_search(new_records,
                        channels=channels,
                        experiments=args.experiments,
                        deepest_net_size=args.deep)


if __name__ == "__main__":
    main()
