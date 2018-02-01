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


def lr_search(experiments=10,
              records=None):
    """
    :type train_data_path: str
    :type eval_data_path: str
    """
    LR = np.random.random_sample([experiments]) / 100
    LR = np.append(LR, 0.02)
    experiments += 1
    LR.sort()
    numeric_result = []
    results = []
    info = []
    LR = list(LR)

    for lr in LR:
        config = Config(architecture=[4],
                        activations=None,
                        batch_size=32,
                        epochs=5,
                        num_steps=5000,
                        save_step=200,
                        learning_rate=lr)

        data = DataHolder(config,
                          records=records)
        name = "lr = {0:.6f}".format(lr)
        print(name + ":\n")
        graph = tf.Graph()
        network = DFN(graph, config)
        trainer = Trainer(graph, config, network, data)
        trainer.fit(verbose=True)
        valid_acc = trainer.get_valid_accuracy()
        numeric_result.append(valid_acc)
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
        attrs = vars(config)
        config_info = ["%s: %s" % item for item in attrs.items()]
        info.append(config_info)

    best_result = max(list(zip(numeric_result, LR, info)))
    result_string = """In an experiment with {0} learning rate values
    the best one is {1} with valid accuracy of {2}.
    \nThe training uses the following params:
    \n{3}\n""".format(experiments,
                      best_result[1],
                      best_result[0],
                      best_result[2])
    file = open("learing_rate.txt", "w")
    file.write("Results with different values for learing_rate\n")
    for result in results:
        result += "\n"
        file.write(result)
    file.write("\n")
    file.write(result_string)
    file.close()


def main():
    """
    Main script.
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('train_data',
    #                     type=str, help='train data path')
    # parser.add_argument('label_data',
    #                     type=str, help='label data path')
    # args = parser.parse_args()
    lr_search(experiments=10,
              records=["pista1_flip_train.tfrecords",
                       "pista1_flip_valid.tfrecords",
                       "pista1_flip_test.tfrecords"])


if __name__ == "__main__":
    main()
