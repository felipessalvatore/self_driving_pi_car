#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import numpy as np

command2int = {"up": 0, "left": 1, "right": 2}
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
    """
    Function to transform a tf records into a tuple of
    np arrays. The size is controled by the param "bound".

    :param record_path: path to tf_record
    :type record_path: str
    :param bound: number of examples to be read
    :type bound: int
    :rtype: np.array, np.array
    """
    reconstructed_images = []
    reconstructed_labels = []
    record_iterator = tf.python_io.tf_record_iterator(path=record_path)

    for i, string_record in enumerate(record_iterator):
        if i <= bound:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            height = int(example.features.feature['height'].int64_list.value[0])  # noqa
            width = int(example.features.feature['width'].int64_list.value[0])  # noqa
            channels = int(example.features.feature['channels'].int64_list.value[0])  # noqa
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
    """
    Function to give the model's accuracy for each category.

    :param pred: model's prediction
    :type pred: np.array
    :param label: true labels
    :type label: np.array
    :rtype: list of float
    """
    pred, label = list(pred), list(label)
    results = []
    for cat in range(categories):
        vfunc = np.vectorize(lambda x: 1 if x == cat else 0)
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
                                            categories=3,
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


def parser_with_normalization(tfrecord):
    """
    Parser function, transforming string into
    a tuple of tensors.

    :param tfrecord: a single binary serialized
    :type tfrecord: tf.Tensor(shape=(), dype=tf.string)
    :rtype: tf.Tensor(shape=(1, height*width*channels),
                      dtype=tf.float32),
            tf.Tensor(shape=(1,), dtyṕe=tf.int32)
    """
    features = {'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'labels_raw': tf.FixedLenFeature([], tf.string)}

    tfrecord_parsed = tf.parse_single_sequence_example(
        tfrecord, features)

    image = tf.decode_raw(tfrecord_parsed[0]['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) / 255

    label = tf.decode_raw(tfrecord_parsed[0]['labels_raw'], tf.uint8)
    label = tf.cast(label, tf.int32)

    return image, label


def get_iterator(filename, batch_size, parser):
    """
    Function to get an interator.

    :param filename: path to tfrecord dataset
    :type filename: str
    :param batch_size: size of the batch
    :type batch_size: int
    :param parser: function to parse a string
                   into a tensor
    :type parser: tf.Tensor(shape=(), dype=tf.string)
                  ->
                  tf.Tensor(shape=(1, height*width*channels),
                      dtype=tf.float32),
                  tf.Tensor(shape=(1,), dtyṕe=tf.int32)
    :rtype: tf.contrib.data.Iterator
    """
    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(batch_size * 2)
    iterator = dataset.make_initializable_iterator()
    return iterator
