from tfnet.base_network import NeuralNetwork
from tfnet.layers import conv2d, max_pool_2x2, conv_bn_relu, flatten, fc_bn_relu, dropout, linear

import tensorflow as tf


class ColorNet(NeuralNetwork):

    def __init__(self):
        x_shape = [None, 32, 32, 1]
        y_shape = [None, 32, 32, 3]
        NeuralNetwork.__init__(self, x_shape, y_shape, name='colornet')

    def define_hyperparams(self):
        self.dropout = tf.placeholder(tf.float32, name='dropout')

    def define_network(self):
        conv_1 = tf.nn.relu(conv2d(self.x, [3, 3, 1, 64], 'C1'))
        conv_2 = tf.nn.relu(conv2d(conv_1, [3, 3, 64, 64], 'C2'))
        conv_3 = tf.nn.relu(conv2d(conv_2, [3, 3, 64, 64], 'C3'))
        drop_3 = dropout(conv_3, self.dropout)
        conv_4 = tf.nn.relu(conv2d(drop_3, [3, 3, 64, 128], 'C4'))
        conv_5 = tf.nn.relu(conv2d(conv_4, [3, 3, 128, 128], 'C5'))
        conv_6 = tf.nn.relu(conv2d(conv_5, [3, 3, 128, 128], 'C6'))
        drop_6 = dropout(conv_6, self.dropout)
        conv_7 = tf.nn.relu(conv2d(drop_6, [3, 3, 128, 256], 'C7'))
        conv_8 = tf.nn.relu(conv2d(conv_7, [3, 3, 256, 256], 'C8'))
        conv_9 = tf.nn.relu(conv2d(conv_8, [3, 3, 256, 256], 'C9'))
        drop_9 = dropout(conv_9, self.dropout)
        output = tf.nn.sigmoid(conv2d(drop_9, [3, 3, 256, 3], 'output'))
        return output

    def define_loss(self):
        loss = tf.reduce_sum(
            tf.pow(self._network - self.y_, 2))
        # add l2 regularization loss
        #for t in tf.trainable_variables():
        #    if 'bias' not in t.name:
        #        loss += tf.nn.l2_loss(t)*1e-5
        return loss

    def define_optimizer(self):
        return tf.train.AdadeltaOptimizer(1.)\
            .minimize(self._loss)

    def set_learning_rate(self, lr):
        if self._session_running:
            self._session.run(self._learning_rate.assign(lr))
        else:
            print('learning rate can only be assigned in a running session.')

    def predict(self, feed_dict={}):
        return self._session.run(tf.nn.top_k(tf.nn.softmax(self._network), 3),
                                 feed_dict=feed_dict)

    @property
    def learning_rate(self):
        if self._session_running:
            return self._session.run(self._learning_rate)
        else:
            print('learning rate can only be evaluated in session.')
            return None
