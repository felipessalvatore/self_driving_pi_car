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
from CNN import CNN
from util import reconstruct_from_record, accuracy_per_category
from util import int2command


def lr_search(name_tfrecords,
              records,
              height,
              width,
              channels,
              architecture,
              activations,
              conv_architecture,
              kernel_sizes,
              pool_kernel,
              batch_size,
              epochs,
              num_steps,
              save_step,
              optimizer,
              experiments,
              conv,
              divisor):
    """
    Script to run different experiments
    to search a learning rate value,
    the result is saved on the file learning_rate_results.txt

    :param name_tfrecords: name of the used tfrecords
    :type name_tfrecords: str
    :param records: list of paths to train, test, and valid tfrecords
    :type records: list of str
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param architecture: network architecture
    :type architecture: list of int
    :param activations: list of different tf functions
    :type activations: list of tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh
    :param conv_architecture: convolutional architecture
    :type conv_architecture: list of int
    :param kernel_sizes: filter sizes
    :type kernel_sizes: list of int
    :param pool_kernel: pooling filter sizes
    :type pool_kernel: list of int
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param num_steps: number of iterations for each epoch
    :type num_steps: int
    :param save_step: when step % save_step == 0, the model
                      parameters are saved.
    :type save_step: int
    :param optimizer: a optimizer from tensorflow.
    :type optimizer: tf.train.GradientDescentOptimizer,
                     tf.train.AdadeltaOptimizer,
                     tf.train.AdagradOptimizer,
                     tf.train.AdagradDAOptimizer,
                     tf.train.AdamOptimizer,
                     tf.train.FtrlOptimizer,
                     tf.train.ProximalGradientDescentOptimizer,
                     tf.train.ProximalAdagradOptimizer,
                     tf.train.RMSPropOptimizer
    :param experiments: number of experiments to be made
    :type experiments: int
    :param conv: param to control if the model will be a CNN
                 or DFN
    :type conv: bool
    :param divisor: param to resize the learning rate
    :type divisor: float
    """
    LR = np.random.random_sample([experiments]) / divisor
    LR.sort()
    numeric_result = []
    results = []
    info = []
    LR = list(LR)
    if conv:
        net_name = "CNN"
    else:
        net_name = "DFN"

    header = "\nSearching learning rate for the model {} in the {} data\n".format(net_name,  # noqa
                                                                                  name_tfrecords)  # noqa
    print(header)
    for lr in LR:
        config = Config(height=height,
                        width=width,
                        channels=channels,
                        learning_rate=lr,
                        architecture=architecture,
                        activations=activations,
                        conv_architecture=conv_architecture,
                        kernel_sizes=kernel_sizes,
                        pool_kernel=pool_kernel,
                        batch_size=batch_size,
                        epochs=epochs,
                        num_steps=num_steps,
                        save_step=save_step,
                        optimizer=optimizer)

        data = DataHolder(config,
                          records=records)
        name = "lr = {0:.6f}".format(lr)
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
        test_images, test_labels, _ = reconstruct_from_record(data.get_test_tfrecord())  # noqa
        test_images = test_images.astype(np.float32) / 255
        test_pred = trainer.predict(test_images)
        acc_cat = accuracy_per_category(test_pred, test_labels, categories=3)
        for i, cat_result in enumerate(acc_cat):
            name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
        results.append(name)
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        info.append(str(config))

    best_result = max(list(zip(numeric_result, LR, info)))
    result_string = """In an experiment with {0} learning rate values
    the best one is {1} with valid accuracy of {2}.
    \nThe training uses the following params:
    \n{3}\n""".format(experiments,
                      best_result[1],
                      best_result[0],
                      best_result[2])
    file = open("learning_rate_results.txt", "w")
    file.write(header)
    file.write("Results with different values for learning rate\n")
    for result in results:
        result += "\n"
        file.write(result)
    file.write("\n")
    file.write(result_string)
    file.close()


def main():
    """
    Main script to perform learnig rate search.
    """
    parser = argparse.ArgumentParser(description='Perform learnig rate search')
    parser.add_argument("-n",
                        "--name_tfrecords",
                        type=str,
                        default="data",
                        help="name for tfrecords (default=data)")  # noqa
    parser.add_argument("-ex",
                        "--experiments",
                        type=int,
                        default=10,
                        help="number of experiments")
    parser.add_argument('-a',
                        '--architecture',
                        type=int,
                        nargs='+',
                        help='sizes for hidden layers and output layer, should end with at least "3" !, (default=[3])',  # noqa
                        default=[3])
    parser.add_argument('-ac',
                        '--activations',
                        type=str,
                        nargs='+',
                        help='activations: relu, sigmoid, tanh (defaul=None)',
                        default=None)
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
    parser.add_argument("-c",
                        "--channels",
                        type=int,
                        default=3,
                        help="number of channels (default=3)")
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
    parser.add_argument("-di",
                        "--divisor",
                        type=float,
                        default=100.0,
                        help="value to divide the learning rate array (default=100.0)")  # noqa
    args = parser.parse_args()
    records = ["_train.tfrecords", "_valid.tfrecords", "_test.tfrecords"]
    new_records = []
    for record in records:
        record = args.name_tfrecords + record
        new_records.append(record)

    optimizer_dict = {"GradientDescent": tf.train.GradientDescentOptimizer,  # noqa
                      "Adadelta": tf.train.AdadeltaOptimizer,
                      "Adagrad": tf.train.AdagradOptimizer,
                      "Adam": tf.train.AdamOptimizer,
                      "Ftrl": tf.train.FtrlOptimizer,
                      "ProximalGradientDescent": tf.train.ProximalGradientDescentOptimizer,  # noqa
                      "ProximalAdagrad": tf.train.ProximalAdagradOptimizer,  # noqa
                      "RMSProp": tf.train.RMSPropOptimizer}  # noqa

    activations_dict = {"relu": tf.nn.relu,
                        "sigmoid": tf.nn.sigmoid,
                        "tanh": tf.nn.tanh}
    if args.activations is not None:
        activations = [activations_dict[act] for act in args.activations]
    else:
        activations = args.activations
    optimizer = optimizer_dict[args.optimizer]

    lr_search(name_tfrecords=args.name_tfrecords,
              records=new_records,
              height=args.height,
              width=args.width,
              channels=args.channels,
              experiments=args.experiments,
              architecture=args.architecture,
              activations=activations,
              conv_architecture=args.conv_architecture,
              kernel_sizes=args.kernel_sizes,
              pool_kernel=args.pool_kernel,
              batch_size=args.batch_size,
              epochs=args.epochs,
              num_steps=args.num_steps,
              save_step=args.save_step,
              optimizer=optimizer,
              conv=args.conv,
              divisor=args.divisor)


if __name__ == "__main__":
    main()
