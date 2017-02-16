
import numpy as np
import tensorflow as tf

from neural_network.layers import conv_bn_relu, max_pool_2x2, flatten, fc_bn_relu, dropout

class NeuralNetwork(object):
	def __init__(self):
		self.weight_variables = []
		self.bias_variables = []

		self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
		self._network = None
		self._loss = None
		self._optimizer = None
		self._accuracy = None
		self._learning_rate = tf.Variable(1e-3)
		self._dropout_prob = 0.5

		self._build_network()
		self._session = None

	def _build_network(self):
		try:
			self._network = self.network_definition()
		except NotImplementedError as e:
			print('Error: network_definition() is not implemented.')
		try:
			self._loss = self.loss_definition()
		except NotImplementedError as e:
			print('Error: loss_definition() is not implemented.')
		try:
			self._optimizer = self.optimizer_definition()
		except NotImplementedError as e:
			print('Error: optimizer_definition() is not implemented.')
		try:
			self.load_hyper_parameters()
		except NotImplementedError as e:
			print('load_hyper_parameters() is not implemented, using default params.')

		correct_prediction = tf.equal(tf.argmax(self._network, 1), tf.argmax(self.y_, 1))
		self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def start_session(self):
		self._session = tf.Session()
		self._session.run(tf.global_variables_initializer())

	def end_session(self):
		self._session.close()

	def train_batch(self, x_batch, y_batch):
		self._session.run(self._optimizer, feed_dict=
						 {self.x: x_batch, self.y_: y_batch, self.dropout_keep_prob: self._dropout_prob})

	def predict(self, x_batch):
		return self._session.run(self._network, feed_dict=
						 {self.x: x_batch, self.dropout_keep_prob: 1.})

	def accuracy(self, x_batch, y_batch):
		return self._session.run(self._accuracy, feed_dict=
			{self.x: x_batch, self.y_: y_batch, self.dropout_keep_prob: 1.})

	def network_definition(self):
		raise NotImplementedError

	def loss_definition(self):
		raise NotImplementedError

	def optimizer_definition(self):
		raise NotImplementedError

	def load_hyper_parameters(self):
		raise NotImplementedError


