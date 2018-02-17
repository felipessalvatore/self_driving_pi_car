#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers import pooling


class CNN():
    """
    A general Convolutional neural network

    :param graph: computation graph
    :type graph: tf.Graph
    :param config:  config class holding info about the
                    number of hidden layers (size of the list)
                    and the number of neurons in each
                    layer (number in the list), and
                    the different activation functions.

    :type config: Config
    """
    def __init__(self,
                 graph,
                 config):
        self.activations = config.activations
        self.architecture = config.architecture
        if self.activations is not None:
            assert len(self.architecture) - 1 == len(self.activations)
        self.conv_architecture = config.conv_architecture
        self.kernel_sizes = config.kernel_sizes
        if self.kernel_sizes is None:
            self.kernel_sizes = [5] * len(self.conv_architecture)
        assert len(self.kernel_sizes) == len(self.conv_architecture)
        self.pool_kernel = config.pool_kernel
        if self.pool_kernel is None:
            self.pool_kernel = [2] * len(self.conv_architecture)
        assert len(self.pool_kernel) == len(self.conv_architecture)
        self.graph = graph
        self.height = config.height
        self.width = config.width
        self.channels = config.channels
        with self.graph.as_default():
            self.build_net()

    def build_net(self, kernel_init=None):
        """
        Build network layers.

        :param kernel_init: variable initializer
        :type kernel_init: None or tf.contrib.layers.xavier_initializer
        """
        self.layers = []
        self.conv_layers = []
        self.pool_layers = []
        architecture_size = len(self.architecture)
        for i, units in enumerate(self.conv_architecture):
            kernel_size = [self.kernel_sizes[i], self.kernel_sizes[i]]
            conv_kernel_size = [self.pool_kernel[i],
                                self.pool_kernel[i]]
            conv = tf.layers.Conv2D(filters=units,
                                    kernel_size=kernel_size,
                                    padding='SAME',
                                    activation=tf.nn.relu)
            self.conv_layers.append(conv)
            pool = pooling.MaxPool2D(pool_size=conv_kernel_size,
                                     strides=2)
            self.pool_layers.append(pool)
        for i, units in enumerate(self.architecture):
            if i != architecture_size - 1:
                if self.activations is None:
                    activation = tf.nn.relu
                else:
                    activation = self.activations[i]
                layer = tf.layers.Dense(units=units,
                                        activation=activation,
                                        kernel_initializer=kernel_init,
                                        name="layer" + str(i + 1))
                self.layers.append(layer)
            else:
                layer = tf.layers.Dense(units=units,
                                        activation=None,
                                        kernel_initializer=kernel_init,
                                        name="output_layer")
                self.layers.append(layer)

    def get_logits(self, img_input):
        """
        Get logits from img_input.

        :param img_input: input image
        :type img_input: tf.Tensor(shape=(None,height*width*channels),
                                          dype=tf.float32)
        :rtype: tf.Tensor(shape=(None, categories),
                          dype=tf.float32)
        """
        with self.graph.as_default():
            with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
                img_input = tf.reshape(img_input,
                                       [-1, self.height,
                                        self.width,
                                        self.channels])
                # Convolutional and Pooling Layers
                for conv, pool in zip(self.conv_layers, self.pool_layers):
                    img_input = conv(img_input)
                    img_input = pool(img_input)
                # Reshaping
                shape = img_input.get_shape()
                flat_shape = int(shape[1] * shape[2] * shape[3])
                tf_input = tf.reshape(img_input, (-1, flat_shape))
                # Dense Layers
                for layer in self.layers:
                    tf_input = layer(tf_input)
        return tf_input

    def get_prediction(self, img_input):
        """
        Get prediction from img_input.

        :param img_input: input image
        :type img_input: tf.Tensor(shape=(None,height*width*channels),
                                          dype=tf.float32)
        :rtype: tf.Tensor(shape=(None, categories),
                          dype=tf.float32)
        """
        with self.graph.as_default():
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                logits = self.get_logits(img_input)
                softmax = tf.nn.softmax(logits, name="output_layer_softmax")
        return softmax
