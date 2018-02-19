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
from CNN import CNN
from util import reconstruct_from_record
from util import int2command


almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from plot.util import plotconfusion  # noqa


def acc(name_tfrecords,
        records,
        height,
        width,
        channels,
        architecture,
        activations,
        conv_architecture,
        kernel_sizes,
        pool_kernel,
        test,
        name,
        conv):
    """
    Checks model's accuracy

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
    :param test: param to control if the test accuracy will be printed.
    :type test: bool
    :param name: name to save the confusion matrix plot.
    :type name: str
    :param conv: param to control if the model will be a CNN
                 or DFN
    :type conv: bool
    """

    config = Config(height=height,
                    width=width,
                    channels=channels,
                    architecture=architecture,
                    activations=activations,
                    conv_architecture=conv_architecture,
                    kernel_sizes=kernel_sizes,
                    pool_kernel=pool_kernel)

    data = DataHolder(config,
                      records=records)

    graph = tf.Graph()
    if conv:
        net_name = "CNN"
        network = CNN(graph, config)
    else:
        net_name = "DFN"
        network = DFN(graph, config)
    trainer = Trainer(graph, config, network, data)
    print("\nAccuracy of the {} model in the {} data\n".format(net_name,
                                                               name_tfrecords))
    print("params:\n{}\n".format(str(config)))
    if not os.path.exists("checkpoints"):
        print("===Accuracy of a non trained model===")

    valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord(), bound=10000)  # noqa
    valid_images = valid_images.astype(np.float32) / 255
    valid_pred = trainer.predict(valid_images)
    valid_labels = valid_labels.reshape((valid_labels.shape[0],))
    plotconfusion(valid_labels,
                  valid_pred,
                  name + "_valid.png",
                  int2command,
                  classes=["left", "right", "up"])

    if test:
        test_images, test_labels, _ = reconstruct_from_record(data.get_test_tfrecord(), bound=10000)  # noqa
        test_images = test_images.astype(np.float32) / 255
        test_pred = trainer.predict(test_images)
        test_labels = test_labels.reshape((test_labels.shape[0],))
        plotconfusion(test_labels,
                      test_pred,
                      name + "_test.png",
                      int2command,
                      classes=["left", "right", "up"])


def main():
    """
    Main script to check model's accuracy using one kind of data.
    """
    parser = argparse.ArgumentParser(description="Checks model's accuracy")
    parser.add_argument("-n",
                        "--name_tfrecords",
                        type=str,
                        default="data",
                        help="name for tfrecords (default=data)")  # noqa
    parser.add_argument("-c",
                        "--channels",
                        type=int,
                        default=3,
                        help="number of channels (default=3)")
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
    parser.add_argument('-conva',
                        '--conv_architecture',
                        type=int,
                        nargs='+',
                        help='filters for conv layers (default=[32, 64])',
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
    parser.add_argument("-t",
                        "--test",
                        action="store_true",
                        default=False,
                        help="print test results and calculate confusion matrix (default=False)")  # noqa
    parser.add_argument("-cm",
                        "--cm_name",
                        type=str,
                        default="Confusion_Matrix",
                        help="name to save confusion matrix plot (default=Confusion_Matrix)")  # noqa
    parser.add_argument("-conv",
                        "--conv",
                        action="store_true",
                        default=False,
                        help="Use convolutional network (default=False)")
    args = parser.parse_args()
    records = ["_train.tfrecords", "_valid.tfrecords", "_test.tfrecords"]
    new_records = []
    for record in records:
        record = args.name_tfrecords + record
        new_records.append(record)

    activations_dict = {"relu": tf.nn.relu,
                        "sigmoid": tf.nn.sigmoid,
                        "tanh": tf.nn.tanh}
    if args.activations is not None:
        activations = [activations_dict[act] for act in args.activations]
    else:
        activations = args.activations

    acc(name_tfrecords=args.name_tfrecords,
        records=new_records,
        height=args.height,
        width=args.width,
        channels=args.channels,
        architecture=args.architecture,
        activations=activations,
        conv_architecture=args.conv_architecture,
        kernel_sizes=args.kernel_sizes,
        pool_kernel=args.pool_kernel,
        test=args.test,
        name=args.cm_name,
        conv=args.conv)


if __name__ == '__main__':
    main()
