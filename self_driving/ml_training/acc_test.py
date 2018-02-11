import tensorflow as tf
import os
import numpy as np
import argparse
import sys
import inspect

from DataHolder import DataHolder
from Config import Config
from Trainer import Trainer
from DFN import DFN
from util import reconstruct_from_record
from util import int2command


almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from plot.util import plotconfusion # noqa


def acc(mode,
        records,
        height,
        width,
        channels,
        architecture,
        activations,
        test,
        name):
    """
    Checks model's accuracy

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
    :param test: param to control if the test accuracy will be printed.
    :type test: bool
    :param name: name to save the confusion matrix plot.
    :type name: str
    """

    config = Config(height=height,
                    width=width,
                    channels=channels,
                    architecture=architecture,
                    activations=activations)

    data = DataHolder(config,
                      records=records)

    graph = tf.Graph()
    network = DFN(graph, config)
    trainer = Trainer(graph, config, network, data)
    print("\nAccuracy in the {} data\n".format(mode))
    print("params:\n{}\n".format(config.get_status()))
    if not os.path.exists("checkpoints"):
        print("===Accuracy of a non trained model===")

    valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord(), bound=10000) # noqa
    valid_images = valid_images.astype(np.float32) / 255
    valid_pred = trainer.predict(valid_images)
    valid_labels = valid_labels.reshape((valid_labels.shape[0],))
    plotconfusion(valid_labels, valid_pred, name + "_valid.png", int2command, classes=["left", "right", "up"]) # noqa

    if test:
        test_images, test_labels, _ = reconstruct_from_record(data.get_test_tfrecord(), bound=10000) # noqa
        test_images = test_images.astype(np.float32) / 255
        test_pred = trainer.predict(test_images)
        test_labels = test_labels.reshape((test_labels.shape[0],))
        plotconfusion(test_labels, test_pred, name + "_test.png", int2command, classes=["left", "right", "up"]) # noqa


def main():
    """
    Main script to check model's accuracy using one kind of data.

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
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="bin",
                        help="mode for data: pure, flip, aug, bin, gray, green (default=pure)")  # noqa
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
    parser.add_argument('-a',
                        '--architecture',
                        type=int,
                        nargs='+',
                        help='sizes for hidden layers and output layer, should end with least "3" !, (default=[3])',  # noqa
                        default=[3])
    parser.add_argument('-ac',
                        '--activations',
                        type=str,
                        nargs='+',
                        help='activations: relu, sigmoid, tanh (defaul=None)',
                        default=None)
    parser.add_argument("-t",
                        "--test",
                        action="store_true",
                        default=False,
                        help="print test results and calculate confusion matrix (default=False)")  # noqa
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="Confusion_Matrix",
                        help="name to save confusion matrix plot (default=Confusion_Matrix)")  # noqa
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

    activations_dict = {"relu": tf.nn.relu,
                        "sigmoid": tf.nn.sigmoid,
                        "tanh": tf.nn.tanh}
    if args.activations is not None:
        activations = [activations_dict[act] for act in args.activations]
    else:
        activations = args.activations

    acc(mode=args.mode,
        records=new_records,
        height=args.height,
        width=args.width,
        channels=channels,
        architecture=args.architecture,
        activations=activations,
        test=args.test,
        name=args.name)


if __name__ == '__main__':
    main()
