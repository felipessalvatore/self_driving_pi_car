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
                 architecture=[1000, 500, 100, 10, 4],
                 batch_size=32,
                 epochs=10,
                 num_steps=1000,
                 save_step=100,
                 learning_rate=0.00217346380124,
                 optimizer=tf.train.GradientDescentOptimizer):
        self.height = height
        self.width = width
        self.channels = channels
        self.architecture = architecture
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_steps = num_steps
        self.save_step = save_step
        self.learning_rate = learning_rate
        self.optimizer = optimizer
