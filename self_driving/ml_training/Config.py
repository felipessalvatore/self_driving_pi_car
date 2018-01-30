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
    :param architecture: network architecture
    :type architecture: list of int
    :param activations: list of different tf functions
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
    :type early_stopping: int
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
                 architecture=[4],
                 activations=None,
                 batch_size=32,
                 epochs=5,
                 num_steps=1000,
                 save_step=100,
                 learning_rate=0.02,
                 optimizer=tf.train.GradientDescentOptimizer):
        self.height = height
        self.width = width
        self.channels = channels
        self.architecture = architecture
        self.activations = activations
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_steps = num_steps
        self.save_step = save_step
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def get_status(self):
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "architecture = {}\n".format(self.architecture)
        status += "activations = {}\n".format(self.activations)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "num_steps = {}\n".format(self.num_steps)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "optimizer = {}\n".format(self.optimizer)
        return status
