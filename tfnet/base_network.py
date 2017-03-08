
import numpy as np
import tensorflow as tf
import os

from tfnet.layers import conv_bn_relu, max_pool_2x2, flatten, fc_bn_relu, dropout


class NeuralNetwork(object):
    def __init__(self, x_shape, y_shape, name='default_name'):
        self.name = name
        self.x = tf.placeholder(tf.float32, shape=x_shape, name='x')
        self.y_ = tf.placeholder(tf.float32, shape=y_shape, name='y_')

        self._network = None
        self._loss = None
        self._optimizer = None
        self._accuracy = None

        self._build_network()

        self._session = None
        self._session_running = False

        self._saver = None

    def _build_network(self):
        try:
            self.define_hyperparams()
        except NotImplementedError as e:
            print('Warning: define_hyperparams() is not implemented, using default params')
        try:
            self._network = self.define_network()
        except NotImplementedError as e:
            print('Error: network_definition() is not implemented.')
        try:
            self._loss = self.define_loss()
        except NotImplementedError as e:
            print('Error: loss_definition() is not implemented.')
        try:
            self._optimizer = self.define_optimizer()
        except NotImplementedError as e:
            print('Error: optimizer_definition() is not implemented.')

    def start_session(self):
        self._saver = tf.train.Saver()
        self._session = tf.Session()
        self._session_running = True
        self._session.run(tf.global_variables_initializer())

    def load_parameters(self, path):
        self._saver.restore(self._session, path)

    def end_session(self):
        self._session.close()
        self._session_running = False

    def train_batch(self, feed_dict={}):
        _, loss = self._session.run([self._optimizer,
                                     self._loss],
                                     feed_dict=feed_dict)
        return loss

    def output(self, feed_dict={}):
        return self._session.run(self._network, feed_dict=feed_dict)

    def loss(self, feed_dict={}):
        return self._session.run(self._loss, feed_dict=feed_dict)

    def save(self, filename=None):
        if filename is None:
            filename = self.name
        cwd = os.getcwd()
        save_path = os.path.join(cwd, 'saved_models/%s.ckpt' % filename)
        _ = self._saver.save(self._session, save_path)

    def network_definition(self):
        raise NotImplementedError

    def loss_definition(self):
        raise NotImplementedError

    def optimizer_definition(self):
        raise NotImplementedError

    def load_hyper_parameters(self):
        raise NotImplementedError


