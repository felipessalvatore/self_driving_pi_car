import tensorflow as tf
import os
import numpy as np
import shutil
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


def train(mode,
          records,
          height,
          width,
          channels,
          architecture,
          activations,
          batch_size,
          epochs,
          num_steps,
          save_step,
          learning_rate,
          optimizer,
          verbose=True):

    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")

    config = Config(height=height,
                    width=width,
                    channels=channels,
                    architecture=architecture,
                    activations=activations,
                    batch_size=batch_size,
                    epochs=epochs,
                    num_steps=num_steps,
                    save_step=save_step,
                    learning_rate=learning_rate,
                    optimizer=optimizer)

    data = DataHolder(config,
                      records=records)

    graph = tf.Graph()
    network = DFN(graph, config)
    trainer = Trainer(graph, config, network, data)
    trainer.fit(verbose=verbose)
    if verbose:
        print("\nResults for the training in the {} data\n".format(mode))
        print("The training has used the following params:\n{}\n".format(config.get_status()))  # noqa
        valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord()) # noqa
        valid_images = valid_images.astype(np.float32) / 255
        valid_pred = trainer.predict(valid_images)
        valid_labels = valid_labels.reshape((valid_labels.shape[0],))
        plotconfusion(valid_labels, valid_pred, "Confusion_Matrix.png", int2command) # noqa


if __name__ == '__main__':
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
                        "--w",
                        type=int,
                        default=90,
                        help="image height (default=90)")  # noqa
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
    train(mode="bin",
          records=new_records,
          height=90,
          width=160,
          channels=channels,
          architecture=[4],
          activations=None,
          batch_size=32,
          epochs=5,
          num_steps=1000,
          save_step=100,
          learning_rate=0.02,
          optimizer=tf.train.GradientDescentOptimizer,
          verbose=True)

      # def __init__(self,
      #            height=90,
      #            width=160,
      #            channels=3,
      #            architecture=[4],
      #            activations=None,
      #            batch_size=32,
      #            epochs=5,
      #            num_steps=1000,
      #            save_step=100,
      #            learning_rate=0.02,
      #            optimizer=tf.train.GradientDescentOptimizer):

# name = ""
# results = []
# valid_acc = trainer.get_valid_accuracy()
# name += ': valid_acc = {0:.6f} | '.format(valid_acc)
# valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord()) # noqa
# valid_images = valid_images.astype(np.float32) / 255
# valid_pred = trainer.predict(valid_images)
# acc_cat = accuracy_per_category(valid_pred, valid_labels, categories=4)
# for i, cat_result in enumerate(acc_cat):
#     name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
# del acc_cat[1]
# name += "\n mean = {0:.6f} | std = {1:.6f}".format(np.mean(acc_cat),
#                                                    np.std(acc_cat))
# print(name)
