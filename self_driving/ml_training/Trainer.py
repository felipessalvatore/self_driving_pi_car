import os
import numpy as np
import tensorflow as tf
from tf_function import get_iterator, parser_with_normalization
from DataHolder import DataHolder
from Config import Config
from DFN import DFN


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

    def build_graph(self):
        """
        build tensforflow graph
        """
        flat_size = self.height * self.width * self.channels
        with self.graph.as_default():
            self.input_image = tf.placeholder(tf.float32,
                                              shape=(None, flat_size),
                                              name="input_image")
            self.iterator_train = get_iterator(self.tfrecords_train,
                                               self.batch_size,
                                               parser_with_normalization)
            self.iterator_valid = get_iterator(self.tfrecords_valid,
                                               self.batch_size,
                                               parser_with_normalization)
            train_images, train_labels = self.iterator_train.get_next()
            train_images = tf.reshape(train_images,
                                      (self.batch_size, flat_size))
            train_labels = tf.reshape(train_labels, (self.batch_size,))
            train_logits = self.model.get_logits(train_images)
            tf_train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels, # noqa
                                                                       logits=train_logits) # noqa
            self.tf_train_loss = tf.reduce_mean(tf_train_loss)
            optimizer = self.tf_optimizer(self.learning_rate)
            self.update_weights = optimizer.minimize(self.tf_train_loss)

            valid_images, valid_labels = self.iterator_valid.get_next()
            valid_images = tf.reshape(valid_images,
                                      (self.batch_size, flat_size))
            valid_labels = tf.reshape(valid_labels, (self.batch_size,))
            valid_logits = self.model.get_logits(valid_images)
            valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, # noqa
                                                                       logits=valid_logits) # noqa
            self.tf_valid_loss = tf.reduce_mean(valid_loss)

            self.tf_prediction = self.model.get_prediction(self.input_image)

            self.saver = tf.train.Saver()
            self.save_path = os.path.join(self.save_dir, 'best_validation')

    def fit(self):
        """
        fiting the data
        """
        best_valid_loss = float("inf")
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.iterator_train.initializer)
            sess.run(self.iterator_valid.initializer)
            sess.run(tf.global_variables_initializer())
            show_loss = sess.run(self.tf_train_loss)
            for i in range(self.epochs):
                print("loss =", show_loss)
                for step in range(self.iterations):
                    _, loss = sess.run([self.update_weights,
                                        self.tf_train_loss])
                    show_loss = loss
                    if step % self.show_step == 0:
                        valid_loss = sess.run(self.tf_valid_loss)
                        if valid_loss < best_valid_loss:
                            self.saver.save(sess=sess,
                                            save_path=self.save_path)
                            best_valid_loss = valid_loss
                            print("save", valid_loss)

    def predict(self, img):
        """
        predict the data
        """
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess=sess, save_path=self.save_path)
            feed_dict = {self.input_image: img}
            result = sess.run(self.tf_prediction,
                              feed_dict=feed_dict)
            result = np.argmax(result, axis=1)
        return result


if __name__ == '__main__':
    my_graph = tf.Graph()
    my_config = Config()
    my_data = DataHolder(my_config,
                         data_path="data.npy",
                         label_path="labels.npy",
                         record_path="pista1",
                         flip=True,
                         augmentation=True)
    my_data.create_records()
    network = DFN(my_graph, my_config)
    my_train = Trainer(my_graph, my_config, network, my_data)
