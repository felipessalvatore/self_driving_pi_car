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
        with self.graph.as_default():
            self.build_net()

    def build_net(self, kernel_init=None):
        """
        Build network layers
        """
        self.layers = []
        architecture_size = len(self.architecture)
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
        return logits using the "img_input" as input
        """
        with self.graph.as_default():
            with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
                tf_input = img_input
                for layer in self.layers:
                    tf_input = layer(tf_input)
        return tf_input

    def get_prediction(self, img_input):
        """
        return probabilistic using the "img_input" as input
        """
        with self.graph.as_default():
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                logits = self.get_logits(img_input)
                softmax = tf.nn.softmax(logits, name="output_layer_softmax")
        return softmax
