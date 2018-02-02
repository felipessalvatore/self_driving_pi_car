#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import numpy as np

command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]: i[0] for i in command2int.items()}


def run_test(testClass):
    """
    Function to run all the tests from a class of tests.

    :param testClass: class for testing
    :type testClass: unittest.TesCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def reconstruct_from_record(record_path, bound=1000):
    reconstructed_images = []
    reconstructed_labels = []
    record_iterator = tf.python_io.tf_record_iterator(path=record_path)

    for i, string_record in enumerate(record_iterator):
        if i <= bound:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            height = int(example.features.feature['height'].int64_list.value[0]) # noqa
            width = int(example.features.feature['width'].int64_list.value[0]) # noqa
            channels = int(example.features.feature['channels'].int64_list.value[0]) # noqa
            img_string = (example.features.feature['image_raw']
                                          .bytes_list
                                          .value[0])
            annotation_string = (example.features.feature['labels_raw']
                                        .bytes_list
                                        .value[0])

            reconstructed_img = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_annotation = np.fromstring(annotation_string,
                                                     dtype=np.uint8)
            reconstructed_images.append(reconstructed_img)
            reconstructed_labels.append(reconstructed_annotation)
        else:
            break
    shape = (height, width, channels)
    reconstructed_images = np.array(reconstructed_images)
    reconstructed_labels = np.array(reconstructed_labels)
    return reconstructed_images, reconstructed_labels, shape


def accuracy_per_category(pred, label, categories):
    pred, label = list(pred), list(label)
    results = []
    for cat in range(categories):
        f = lambda x: 1 if x == cat else 0 # noqa
        vfunc = np.vectorize(f)
        mapped_pred = vfunc(pred)
        mapped_labels = vfunc(label)
        right = float(np.dot(mapped_pred, mapped_labels))
        total = np.sum(mapped_labels)
        if total == 0:
            results.append(0.0)
        else:
            results.append((right / total))
    return results


def get_random_architecture_and_activations(network_sizes,
                                            categories=4,
                                            upper_bound=6000):
    """
    Creates a random architecture list and activations list
    using a list of sizes for different networks.

    :param network_sizes: list of network's size
    :type network_sizes: list of int
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :rtype: list of int, list of function tensorflow
    """
    activations_dict = {0: tf.nn.relu,
                        1: tf.nn.sigmoid,
                        2: tf.nn.tanh}
    hidden_layers = []
    activations = []
    lower_bound = categories

    for size in network_sizes:
        hidden_sizes = []
        last = upper_bound
        for _ in range(size):
            if lower_bound < last / 2:
                new_size = np.random.randint(lower_bound, last / 2)
            else:
                new_size = np.random.randint(lower_bound, lower_bound + 1)
            hidden_sizes.append(new_size)
            last = new_size
        hidden_layers.append(hidden_sizes)

    for hidden in hidden_layers:
        activ = np.random.randint(0, 3, len(hidden))
        activ = list(map(lambda x: activations_dict[x], activ))
        activations.append(activ)

    for hidden in hidden_layers:
        hidden.append(categories)

    return hidden_layers, activations
