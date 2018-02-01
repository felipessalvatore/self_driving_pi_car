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


def data_search(data_path,
                label_path):
    """
    :type train_data_path: str
    :type eval_data_path: str
    """
    config_pure = Config(architecture=[4],
                         activations=None,
                         batch_size=32,
                         epochs=1,
                         num_steps=6,
                         save_step=2)
    data_pure = DataHolder(config_pure,
                           data_path=data_path,
                           label_path=label_path,
                           record_path="pista1_pure")

    config_flip = Config(architecture=[4],
                         activations=None,
                         batch_size=32,
                         epochs=1,
                         num_steps=6,
                         save_step=2)
    data_flip = DataHolder(config_flip,
                           data_path=data_path,
                           label_path=label_path,
                           record_path="pista1_flip",
                           flip=True)

    config_aug = Config(architecture=[4],
                        activations=None,
                        batch_size=32,
                        epochs=1,
                        num_steps=6,
                        save_step=2)
    data_aug = DataHolder(config_aug,
                          data_path=data_path,
                          label_path=label_path,
                          record_path="pista1_aug",
                          flip=True,
                          augmentation=True)

    config_bin = Config(architecture=[4],
                        activations=None,
                        batch_size=32,
                        epochs=1,
                        num_steps=6,
                        save_step=2)
    data_bin = DataHolder(config_bin,
                          data_path=data_path,
                          label_path=label_path,
                          record_path="pista1_bin",
                          flip=True,
                          binary=True)

    config_green = Config(architecture=[4],
                          activations=None,
                          batch_size=32,
                          epochs=1,
                          num_steps=6,
                          save_step=2)
    data_green = DataHolder(config_green,
                            data_path=data_path,
                            label_path=label_path,
                            record_path="pista1_green",
                            flip=True,
                            green=True)

    config_gray = Config(architecture=[4],
                         activations=None,
                         batch_size=32,
                         epochs=1,
                         num_steps=6,
                         save_step=2)
    data_gray = DataHolder(config_gray,
                           data_path=data_path,
                           label_path=label_path,
                           record_path="pista1_gray",
                           flip=True,
                           green=True)

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
        data.create_records()
        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph, config, network, data)
        trainer.fit(verbose=True)
        valid_acc = trainer.get_valid_accuracy()
        name += ': valid_acc = {0:.6f} | '.format(valid_acc)
        test_images, test_labels, _ = reconstruct_from_record(data.get_test_tfrecord()) # noqa
        test_images = test_images.astype(np.float32) / 255
        test_pred = trainer.predict(test_images)
        acc_cat = accuracy_per_category(test_pred, test_labels, categories=4)
        for i, cat_result in enumerate(acc_cat):
            name += int2command[i] + ": = {0:.6f}, ".format(cat_result)
        results.append(name)
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")

    file = open("different_data.txt", "w")
    file.write("Results with different data\n")
    for result in results:
        result += "\n"
        file.write(result)
    file.close()


def main():
    """
    Main script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data',
                        type=str, help='train data path')
    parser.add_argument('label_data',
                        type=str, help='label data path')
    args = parser.parse_args()
    data_search(args.train_data,
                args.label_data)


if __name__ == "__main__":
    main()
