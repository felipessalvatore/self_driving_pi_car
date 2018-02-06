#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


class DFN():
    """
    A general Deep feedforward network

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
        self.graph = graph

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
        :rtype: tf.Tensor(shape=(None, categories),
                          dype=tf.float32)
        """
        with self.graph.as_default():
            with tf.variable_scope("logits", reuse=reuse):
                tf_input = img_input
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
