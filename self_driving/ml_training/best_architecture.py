#!/usr/bin/env python
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
from CNN import CNN
from util import reconstruct_from_record, accuracy_per_category
from util import int2command, get_random_architecture_and_activations


def architecture_search(mode,
                        records,
                        height,
                        width,
                        channels,
                        conv_architecture,
                        kernel_sizes,
                        pool_kernel,
                        batch_size,
                        epochs,
                        num_steps,
                        save_step,
                        learning_rate,
                        optimizer,
                        experiments,
                        deepest_net_size,
                        conv):
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
    if conv:
        net_name = "CNN"
    else:
        net_name = "DFN"

    header = "\nSearching {} architecture in the {} data\n".format(net_name,
                                                                   mode)
    print(header)
    for arch, act in zip(hidden_layers, activations):
        config = Config(height=height,
                        width=width,
                        channels=channels,
                        architecture=arch,
                        activations=act,
                        conv_architecture=conv_architecture,
                        kernel_sizes=kernel_sizes,
                        pool_kernel=pool_kernel,
                        batch_size=batch_size,
                        epochs=epochs,
                        num_steps=num_steps,
                        save_step=save_step,
                        learning_rate=learning_rate,
                        optimizer=optimizer)

        data = DataHolder(config,
                          records=records)
        name = str(arch)
        print(name + ":\n")
        graph = tf.Graph()
        if conv:
            network = CNN(graph, config)
        else:
            network = DFN(graph, config)
        trainer = Trainer(graph, config, network, data)
        trainer.fit(verbose=True)
        valid_acc = trainer.get_valid_accuracy()
        numeric_result.append(valid_acc)
        name += ': valid_acc = {0:.6f} | '.format(valid_acc)
        valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord())  # noqa
        valid_images = valid_images.astype(np.float32) / 255
        valid_pred = trainer.predict(valid_images)
        acc_cat = accuracy_per_category(valid_pred, valid_labels, categories=3)
        for i, cat_result in enumerate(acc_cat):
            name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
        results.append(name)
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        info.append(str(config))

    best_result = max(list(zip(numeric_result, hidden_layers, info)))
    result_string = """In an experiment with {0} architectures
    the best one is {1} with valid accuracy of {2}.
    \nThe training uses the following params:
    \n{3}\n""".format(experiments,
                      best_result[1],
                      best_result[0],
                      best_result[2])
    file = open("architecture_results.txt", "w")
    file.write(header)
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
    parser = argparse.ArgumentParser(description='Perform architecture search')
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="pure",
                        help="mode for data: pure, flip, aug, bin, gray, green (default=pure)")  # noqa
    parser.add_argument("-ex",
                        "--experiments",
                        type=int,
                        default=10,
                        help="number of experiments to be done (default=10)")  # noqa
    parser.add_argument("-d",
                        "--deep",
                        type=int,
                        default=4,
                        help="deep of the model (default=4)")
    parser.add_argument("-he",
                        "--height",
                        type=int,
                        default=90,
                        help="image height (default=90)")
    parser.add_argument("-w",
                        "--width",
                        type=int,
                        default=160,
                        help="image width (default=160)")
    parser.add_argument('-conva',
                        '--conv_architecture',
                        type=int,
                        nargs='+',
                        help='filters for conv layers (default=[32, 64])',  # noqa
                        default=[32, 64])
    parser.add_argument('-k',
                        '--kernel_sizes',
                        type=int,
                        nargs='+',
                        help='kernel sizes for conv layers (default=None - 5 for every layer)',  # noqa
                        default=None)
    parser.add_argument('-p',
                        '--pool_kernel',
                        type=int,
                        nargs='+',
                        help='kernel sizes for pooling layers (default=None - 2 for every layer)',  # noqa
                        default=None)
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=32,
                        help="batch size (default=32)")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=5,
                        help="epochs for training (default=5)")
    parser.add_argument("-ns",
                        "--num_steps",
                        type=int,
                        default=1000,
                        help="number of steps for each epoch (default=1000)")
    parser.add_argument("-ss",
                        "--save_step",
                        type=int,
                        default=100,
                        help="number of steps to save variables (default=100)")
    parser.add_argument("-lr",
                        "--learning_rate",
                        type=float,
                        default=0.02,
                        help="learning rate (default=0.02)")
    opt_list = """optimizers: GradientDescent,
                              Adadelta,
                              Adagrad,
                              Adam,
                              Ftrl,
                              ProximalGradientDescent,
                              ProximalAdagrad,
                              RMSProp"""
    parser.add_argument("-o",
                        "--optimizer",
                        type=str,
                        default="GradientDescent",
                        help=opt_list + "(default=GradientDescent)")
    parser.add_argument("-conv",
                        "--conv",
                        action="store_true",
                        default=False,
                        help="Use convolutional network (default=False)")
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

    optimizer_dict = {"GradientDescent": tf.train.GradientDescentOptimizer, # noqa
                      "Adadelta": tf.train.AdadeltaOptimizer,
                      "Adagrad": tf.train.AdagradOptimizer,
                      "Adam": tf.train.AdamOptimizer,
                      "Ftrl": tf.train.FtrlOptimizer,
                      "ProximalGradientDescent": tf.train.ProximalGradientDescentOptimizer, # noqa
                      "ProximalAdagrad": tf.train.ProximalAdagradOptimizer, # noqa
                      "RMSProp":tf.train.RMSPropOptimizer} # noqa
    optimizer = optimizer_dict[args.optimizer]
    architecture_search(mode=args.mode,
                        records=new_records,
                        height=args.height,
                        width=args.width,
                        channels=channels,
                        experiments=args.experiments,
                        deepest_net_size=args.deep,
                        conv_architecture=args.conv_architecture,
                        kernel_sizes=args.kernel_sizes,
                        pool_kernel=args.pool_kernel,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        num_steps=args.num_steps,
                        save_step=args.save_step,
                        learning_rate=args.learning_rate,
                        optimizer=optimizer,
                        conv=args.conv)


if __name__ == "__main__":
    main()
