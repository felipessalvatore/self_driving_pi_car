class DFN():
    """
     to do
     
    :param graph: computation graph
    :type graph: tf.Graph
    :param architecture: a list that gives the number
                        of hidden layers (size of the list)
                        and the number of neurons in each
                        layer (number in the list)
    :type architecture: list of ints 
    """   
    def __init__(self, graph, architecture, activations=None):
        if activations is not None:
            assert len(activations) - 1 == len(architecture)
        self.architecture = architecture
        self.graph = graph
        self.activations = activations
        with self.graph.as_default():
            self.build_net()
        
    def build_net(self, kernel_init=None):
        """
        to do
        """
        self.layers = []
        architecture_size = len(self.architecture)
        for i, units in enumerate(self.architecture):
            if i != architecture_size - 1:
                if self.activations is None:
                     activation = tf.nn.relu
                else:
                     activation = self.activations[i]
                layer =  tf.layers.Dense(units=units,
                                         activation=activation,
                                         kernel_initializer=kernel_init,
                                         name="layer" + str(i + 1))
                self.layers.append(layer)
            else:
                layer =  tf.layers.Dense(units=units,
                                         activation=None,
                                         kernel_initializer=kernel_init,
                                         name="output_layer")
                self.layers.append(layer)

    def get_logits(self, img_input):
        """
        to do
        """
        with self.graph.as_default():
            with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
                tf_input = img_input
                for layer in self.layers:
                    tf_input = layer(tf_input)
        return tf_input

    def get_prediction(self, img_input):
        """
        to do
        """
        with self.graph.as_default():
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                logits = self.get_logits(img_input)
                softmax = tf.nn.softmax(logits, name="output_layer_softmax") 
        return softmax