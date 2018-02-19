#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


class Config(object):
    """
    Holds model hyperparams.

    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param architecture: network dense architecture
    :type architecture: list of int
    :param activations: list of different tf functions
    :param conv_architecture: convolutional architecture
    :type conv_architecture: list of int
    :param kernel_sizes: filter sizes
    :type kernel_sizes: list of int
    :param pool_kernel: pooling filter sizes
    :type pool_kernel: list of int
    :type activations: list of tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param num_steps: number of iterations for each epoch
    :type num_steps: int
    :param save_step: when step % save_step == 0, the model
                      parameters are saved.
    :type save_step: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param optimizer: a optimizer from tensorflow.
    :type optimizer: tf.train.GradientDescentOptimizer,
                     tf.train.AdadeltaOptimizer,
                     tf.train.AdagradOptimizer,
                     tf.train.AdagradDAOptimizer,
                     tf.train.MomentumOptimizer,
                     tf.train.AdamOptimizer,
                     tf.train.FtrlOptimizer,
                     tf.train.ProximalGradientDescentOptimizer,
                     tf.train.ProximalAdagradOptimizer,
                     tf.train.RMSPropOptimizer
    """
    def __init__(self,
                 height=90,
                 width=160,
                 channels=3,
                 architecture=[722, 3],
                 activations=None,
                 conv_architecture=[32, 64],
                 kernel_sizes=[5, 5],
                 pool_kernel=None,
                 batch_size=32,
                 epochs=5,
                 num_steps=1000,
                 save_step=100,
                 learning_rate=0.0054,
                 optimizer=tf.train.GradientDescentOptimizer):
        self.height = height
        self.width = width
        self.channels = channels
        self.architecture = architecture
        self.activations = activations
        self.conv_architecture = conv_architecture
        self.kernel_sizes = kernel_sizes
        self.pool_kernel = pool_kernel
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_steps = num_steps
        self.save_step = save_step
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def __str__(self):
        """
        Get all attributs values.

        :return: all hyperparams as a string
        :rtype: str
        """
        if self.kernel_sizes is None:
            kernel_sizes = [5] * len(self.conv_architecture)
        else:
            kernel_sizes = self.kernel_sizes
        if self.pool_kernel is None:
            pool_kernel = [2] * len(self.conv_architecture)
        else:
            pool_kernel = self.pool_kernel
        if self.activations is None:
            activations = ["relu"] * len(self.architecture)
        else:
            activations = self.activations
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "architecture = {}\n".format(self.architecture)
        status += "activations = {}\n".format(activations)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "conv_architecture = {}\n".format(self.conv_architecture)
        status += "kernel_sizes = {}\n".format(kernel_sizes)
        status += "pool_kernel = {}\n".format(pool_kernel)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "num_steps = {}\n".format(self.num_steps)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "optimizer = {}\n".format(self.optimizer)
        return status
