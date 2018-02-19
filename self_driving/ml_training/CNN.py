#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class CNN():
    """
    A general Convolutional Neural Network (CNN)

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

    def get_logits(self,
                   img_input,
                   reuse=None):
        """
        Get logits from img_input.

        :param img_input: input image
        :type img_input: tf.Tensor(shape=(None,height*width*channels),
                                          dype=tf.float32)
        :param reuse: param to control reuse variables
        :type reuse: None or True
        :return: logits
        :rtype: tf.Tensor(shape=(None, categories),
                          dype=tf.float32)
        """
        with self.graph.as_default():
            with tf.variable_scope("logits", reuse=reuse):
                img_input = tf.reshape(img_input,
                                       [-1, self.height,
                                        self.width,
                                        self.channels])
                # Convolutional and Pooling Layers
                for i, units in enumerate(self.conv_architecture):
                    kernel_size = [self.kernel_sizes[i], self.kernel_sizes[i]]
                    conv_kernel_size = [self.pool_kernel[i],
                                        self.pool_kernel[i]]
                    conv = tf.contrib.layers.conv2d(inputs=img_input,
                                                    num_outputs=units,
                                                    kernel_size=kernel_size,
                                                    padding='SAME',
                                                    activation_fn=tf.nn.relu)
                    img_input = tf.contrib.layers.max_pool2d(inputs=conv,
                                                             kernel_size=conv_kernel_size)  # noqa
                # Reshaping
                shape = img_input.get_shape()
                flat_shape = int(shape[1] * shape[2] * shape[3])
                tf_input = tf.reshape(img_input, (-1, flat_shape))

                # Dense Layers
                architecture_size = len(self.architecture)
                for i, units in enumerate(self.architecture):
                    if i != architecture_size - 1:
                        if self.activations is None:
                            activation = tf.nn.relu
                        else:
                            activation = self.activations[i]
                        tf_input = tf.contrib.layers.fully_connected(inputs=tf_input,  # noqa
                                                                     num_outputs=units,  # noqa
                                                                     activation_fn=activation)  # noqa
                    else:
                        tf_input = tf.contrib.layers.fully_connected(inputs=tf_input,  # noqa
                                                                     num_outputs=units,  # noqa
                                                                     activation_fn=None)  # noqa
            return tf_input
