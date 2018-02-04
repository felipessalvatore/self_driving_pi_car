#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def randomize_in_place(list1, list2, init):
    """
    Function to randomize two lists the same way.
    Usualy this functions is used when list1 = dataset,
    and list2 = labels.

    :type list1: list
    :type list2: list
    :type init: int
    """
    np.random.seed(seed=init)
    np.random.shuffle(list1)
    np.random.seed(seed=init)
    np.random.shuffle(list2)


def data_cut(data, labels, init=0):
    """
    Given the data and the labels this function shuffles them together
    and separetes four fifths of the data (that is why we use the
    variable ff) to be the traing data; the rest of the data
    is divide into valid data and test data. If the size of the
    data is odd we add one observation copy to the dataset.

    :type data: np array
    :type labels: np array
    :type init: int
    :rtype: None or np array
    """
    randomize_in_place(data, labels, init)
    data_size = data.shape[0]
    ff = int((4 / 5) * data_size)
    rest = data_size - ff
    if rest % 2 == 1:
        new_data = data[-1]
        new_label = labels[-1]
        data = np.vstack([data, new_data])
        labels = np.vstack([labels, new_label])
        rest += 1
    rest = int(rest / 2)
    train_data, train_labels = data[0:ff], labels[0:ff]
    valid_data, valid_labels = data[ff: ff + rest], labels[ff:ff + rest]
    ff = ff + rest
    test_data, test_labels = data[ff: ff + rest], labels[ff: ff + rest]
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels  # noqa


def create_record(record_path,
                  data,
                  labels,
                  height,
                  width,
                  channels):
    """
    Fuction to create one tf.record using two numpy arrays.
    The array in data_path is espected to be flat.

    :param record_path: path to save the tf.record
    :type record_path: str
    :param data_path: path to load the data
    :type data_path: str
    :param label_path: path to load the labels
    :type label_path: str
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    """
    assert data.shape[1] == height * width * channels
    writer = tf.python_io.TFRecordWriter(record_path)
    for i, e in enumerate(data):
        img_str = data[i].tostring()
        label_str = labels[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channels': _int64_feature(channels),
            'image_raw': _bytes_feature(img_str),
            'labels_raw': _bytes_feature(label_str)}))

        writer.write(example.SerializeToString())
    writer.close()
