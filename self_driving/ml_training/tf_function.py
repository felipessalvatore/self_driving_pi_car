#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def parser_with_normalization(tfrecord):
    """
    Parser function to dataset
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
    return iterator from dataset
    """
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(batch_size * 2)
    iterator = dataset.make_initializable_iterator()
    return iterator
