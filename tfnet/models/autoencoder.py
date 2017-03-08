from tfnet.base_network import NeuralNetwork
from tfnet.layers import max_pool_2x2, conv_bn_relu, flatten, fc_bn_relu, dropout, linear

import tensorflow as tf


class CifarAutoEncoder(NeuralNetwork):

    def __init__(self):
        x_shape = [None, 32, 32, 3]
        y_shape = [None, 32, 32, 3]
        NeuralNetwork.__init__(self, x_shape, y_shape, name='autoencoder')

    def define_hyperparams(self):
        pass

    def define_network(self):
        W = tf.Variable(tf.truncated_normal([7, 7, 3, 64], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[64]))

        b_prime = tf.Variable(tf.constant(0.1, shape=[3]))

        hidden = tf.nn.sigmoid(tf.nn.conv2d(self.x, W, strides=[1,1,1,1], padding='SAME') + b)
        output = tf.nn.sigmoid(tf.nn.conv2d_transpose(hidden, W, [128, 32, 32, 3], strides=[1,1,1,1], padding='SAME') + b_prime)
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
