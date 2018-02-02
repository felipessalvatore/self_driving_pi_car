#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from tf_function import get_iterator, parser_with_normalization
from DataHolder import DataHolder
from Config import Config
from DFN import DFN
from util import reconstruct_from_record, accuracy_per_category

class Trainer():
    """
    Class that trains and predicts.

    :type data_path: str
    :type label_path: str
    :type record_path: str
    :type flip: boolean
    :type binarize: boolean
    :type gray: boolean
    :type green: boolean
    :type augmentation: boolean
    """
    def __init__(self,
                 graph,
                 config,
                 model,
                 dataholder,
                 save_dir='checkpoints/'):
        self.tf_optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.iterations = config.num_steps
        self.learning_rate = config.learning_rate
        self.height = config.height
        self.width = config.width
        self.channels = config.channels
        self.show_step = config.save_step
        self.tfrecords_train = dataholder.get_train_tfrecord()
        self.tfrecords_valid = dataholder.get_valid_tfrecord()
        self.tfrecords_test = dataholder.get_test_tfrecord()
        self.graph = graph
        self.model = model
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.build_graph()

    def build_graph(self):
        """
        build tensforflow graph
        """
        flat_size = self.height * self.width * self.channels
        with self.graph.as_default():
            with tf.name_scope("placeholders"):
                self.input_image = tf.placeholder(tf.float32,
                                                  shape=(None, flat_size),
                                                  name="input_image")
            with tf.name_scope("iterators"):
                self.iterator_train = get_iterator(self.tfrecords_train,
                                                   self.batch_size,
                                                   parser_with_normalization)
                self.iterator_valid = get_iterator(self.tfrecords_valid,
                                                   self.batch_size,
                                                   parser_with_normalization)
            with tf.name_scope("prediction"):
                self.tf_prediction = self.model.get_prediction(self.input_image) # noqa

            with tf.name_scope("train_loss"):
                train_images, train_labels = self.iterator_train.get_next()
                train_images = tf.reshape(train_images,
                                          (self.batch_size, flat_size))
                train_labels = tf.reshape(train_labels, (self.batch_size,))
                train_logits = self.model.get_logits(train_images)
                tf_train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels, # noqa
                                                                           logits=train_logits) # noqa
                self.tf_train_loss = tf.reduce_mean(tf_train_loss)

            with tf.name_scope("optimization"):
                optimizer = self.tf_optimizer(self.learning_rate)
                self.update_weights = optimizer.minimize(self.tf_train_loss)

            with tf.name_scope("valid_loss"):
                valid_images, valid_labels = self.iterator_valid.get_next()
                valid_images = tf.reshape(valid_images,
                                          (self.batch_size, flat_size))
                valid_labels = tf.reshape(valid_labels, (self.batch_size,))
                valid_logits = self.model.get_logits(valid_images)
                valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, # noqa
                                                                           logits=valid_logits) # noqa
                self.tf_valid_loss = tf.reduce_mean(valid_loss)

            with tf.name_scope("valid_accuracy"):
                valid_prediction = tf.nn.softmax(valid_logits)
                valid_prediction = tf.argmax(valid_prediction, axis=1)
                valid_prediction = tf.cast(valid_prediction, dtype=tf.int32)
                valid_prediction = tf.equal(valid_prediction,
                                            valid_labels)
                self.valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction,
                                                             'float'),
                                                     name='valid_accuracy')

            with tf.name_scope("saver"):
                self.saver = tf.train.Saver()
                self.save_path = os.path.join(self.save_dir, 'best_validation')

    def get_accuracy(self, iterator_initializer, accuracy_tensor, iterations):
        """
        Method to compute the accuracy of the model's predictions
        on the test dataset
        """
        with tf.Session(graph=self.graph) as sess:
            sess.run(iterator_initializer)
            if os.listdir(self.save_dir) == []:
                sess.run(tf.global_variables_initializer())
            else:
                self.saver.restore(sess=sess, save_path=self.save_path)
            acc = 0
            for _ in range(iterations):
                acc += sess.run(accuracy_tensor)
        return acc / iterations

    def get_valid_accuracy(self,
                           iterations=50):
        """
        Method to compute the accuracy of the model's predictions
        on the test dataset
        """
        return self.get_accuracy(self.iterator_valid.initializer,
                                 self.valid_accuracy,
                                 iterations)

    def fit(self, verbose=True):
        """
        fiting the data
        """
        best_valid_loss = float("inf")
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.iterator_train.initializer)
            sess.run(self.iterator_valid.initializer)
            sess.run(tf.global_variables_initializer())
            show_loss = sess.run(self.tf_train_loss)
            for epoch in range(self.epochs):
                for step in range(self.iterations):
                    _, loss = sess.run([self.update_weights,
                                        self.tf_train_loss])
                    show_loss = loss
                    if step % self.show_step == 0:
                        if verbose:
                            info = 'Epoch {0:5},'.format(epoch + 1)
                            info += ' step {0:5}:'.format(step + 1)
                            info += ' train_loss = {0:.6f} |'.format(show_loss)
                            info += ' valid_loss = {0:.6f}\n'.format(best_valid_loss) # noqa
                            print(info, end='') # noqa
                        valid_loss = sess.run(self.tf_valid_loss)
                        if valid_loss < best_valid_loss:
                            self.saver.save(sess=sess,
                                            save_path=self.save_path)
                            best_valid_loss = valid_loss

    def predict(self, img):
        """
        predict the data
        """
        type_msg = "not in the correct type"
        assert img.dtype == np.float32, type_msg
        with tf.Session(graph=self.graph) as sess:
            if os.listdir(self.save_dir) == []:
                sess.run(tf.global_variables_initializer())
            else:
                self.saver.restore(sess=sess, save_path=self.save_path)
            feed_dict = {self.input_image: img}
            result = sess.run(self.tf_prediction,
                              feed_dict=feed_dict)
            result = np.argmax(result, axis=1)
            result = result.astype(np.int32)
        return result
