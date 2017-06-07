from tfnet.base_network import NeuralNetwork
from tfnet.layers import max_pool_2x2, conv2d, flatten, dropout, linear

import tensorflow as tf


class MnistClassifier(NeuralNetwork):

    def __init__(self):
        x_shape = [None, 28, 28, 1]
        y_shape = [None, 10]
        NeuralNetwork.__init__(self, x_shape, y_shape, name='mnistnet')

    def define_hyperparams(self):
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self._learning_rate = tf.Variable(1e-3, trainable=False)

    def define_network(self):
        conv_1 = tf.nn.relu(conv2d(self.x, [5, 5, 1, 32], 'C_1'))
        pool_1 = max_pool_2x2(conv_1)
        conv_2 = tf.nn.relu(conv2d(pool_1, [5, 5, 32, 64], 'C_2'))
        pool_2 = max_pool_2x2(conv_2)

        # size of input is now 7*7*64
        flattened = flatten(pool_2, [-1, 3136])

        fc1 = tf.nn.relu(linear(flattened, [3136, 512], 'FC1'))
        drop_fc1 = dropout(fc1, self.dropout)

        output = linear(drop_fc1, [512, 10], 'output')
        return output

    def define_loss(self):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self._network, labels=self.y_))
        # add l2 regularization loss
        for t in tf.trainable_variables():
            if 'bias' not in t.name:
                loss += tf.nn.l2_loss(t)*1e-5
        return loss

    def define_optimizer(self):
        # return tf.train.AdamOptimizer(self._learning_rate)
        #     .minimize(self._loss)
        return tf.train.AdadeltaOptimizer(self._learning_rate)\
            .minimize(self._loss)
        # return tf.train.RMSPropOptimizer(self._learning_rate)\
        #     .minimize(self._loss)

    def set_learning_rate(self, lr):
        if self._session_running:
            self._session.run(self._learning_rate.assign(lr))
        else:
            print('learning rate can only be assigned in a running session.')

    def predict(self, feed_dict={}):
        return self._session.run(tf.nn.softmax(self._network),
                                 feed_dict=feed_dict)

    @property
    def learning_rate(self):
        if self._session_running:
            return self._session.run(self._learning_rate)
        else:
            print('learning rate can only be evaluated in session.')
            return None
