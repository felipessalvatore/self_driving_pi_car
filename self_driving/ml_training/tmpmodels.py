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
from util import reconstruct_from_record, accuracy_per_category
from util import int2command


bin_record = ["bin_train.tfrecords",
              "bin_valid.tfrecords",
              "bin_test.tfrecords"]
#almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
#pwd = os.path.dirname(almost_current)

config = Config(channels=1,
                learning_rate=0.00543,
                architecture=[770,4],
                activations=[tf.nn.tanh],
                epochs=5)

data = DataHolder(config,
                  records=bin_record)



#if not bool(set(os.listdir(pwd)).intersection(set(bin_record))):
#        data.create_records() 

graph = tf.Graph()
network = DFN(graph, config)
trainer = Trainer(graph, config, network, data)
trainer.fit(verbose=True)

name = "Bin data with GD"
results = []
valid_acc = trainer.get_valid_accuracy()
name += ': valid_acc = {0:.6f} | '.format(valid_acc)
valid_images, valid_labels, _ = reconstruct_from_record(data.get_valid_tfrecord()) # noqa
valid_images = valid_images.astype(np.float32) / 255
valid_pred = trainer.predict(valid_images)
acc_cat = accuracy_per_category(valid_pred, valid_labels, categories = 4)
for i, cat_result in enumerate(acc_cat):
      name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
results.append(name)
#if os.path.exists("checkpoints"):
#    shutil.rmtree("checkpoints")

file = open("bin_gd_results.txt", "w")
file.write("Results with binary, tanh and gd\n")
for result in results:
    result += "\n"
    file.write(result)
file.close()

