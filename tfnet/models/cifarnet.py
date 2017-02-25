from tfnet.base_network import NeuralNetwork
from tfnet.layers import max_pool_2x2, conv_bn_relu, flatten, fc_bn_relu, dropout, linear

import tensorflow as tf

class CifarNet(NeuralNetwork):
	def __init__(self):
		x_shape = [None, 32, 32, 3]
		y_shape = [None, 10]
		NeuralNetwork.__init__(self, x_shape, y_shape, name='cifarnet')

	def define_hyperparams(self):
		self.dropout = tf.placeholder(tf.float32, name='dropout')
		self._learning_rate = tf.Variable(1e-3, trainable=False)

	def define_network(self):
		conv_1_1 = conv_bn_relu(self.x, [3, 3, 3, 64], 'C_1_1')
		drop_1 = dropout(conv_1_1, self.dropout)
		conv_1_2 = conv_bn_relu(drop_1, [3, 3, 64, 64], 'C_1_2')
		pool_1 = max_pool_2x2(conv_1_2)
		
		conv_2_1 = conv_bn_relu(pool_1, [3, 3, 64, 128], 'C_2_1')
		drop_2 = dropout(conv_2_1, self.dropout)
		conv_2_2 = conv_bn_relu(drop_2, [3, 3, 128, 128], 'C_2_2')
		pool_2 = max_pool_2x2(conv_2_2)

		conv_3_1 = conv_bn_relu(pool_2, [3, 3, 128, 256], 'C_3_1')
		drop_3_1 = dropout(conv_3_1, self.dropout)
		conv_3_2 = conv_bn_relu(drop_3_1, [3, 3, 256, 256], 'C_3_2')
		drop_3_2 = dropout(conv_3_2, self.dropout)
		conv_3_3 = conv_bn_relu(drop_3_2, [3, 3, 256, 256], 'C_3_3')
		pool_3 = max_pool_2x2(conv_3_3)

		conv_4_1 = conv_bn_relu(pool_3, [3, 3, 256, 512], 'C_4_1')
		drop_4_1 = dropout(conv_4_1, self.dropout)
		conv_4_2 = conv_bn_relu(drop_4_1, [3, 3, 512, 512], 'C_4_2')
		drop_4_2 = dropout(conv_4_2, self.dropout)
		conv_4_3 = conv_bn_relu(drop_4_2, [3, 3, 512, 512], 'C_4_3')
		pool_4 = max_pool_2x2(conv_4_3)

		conv_5_1 = conv_bn_relu(pool_4, [3, 3, 512, 512], 'C_5_1')
		drop_5_1 = dropout(conv_5_1, self.dropout)
		conv_5_2 = conv_bn_relu(drop_5_1, [3, 3, 512, 512], 'C_5_2')
		drop_5_2 = dropout(conv_5_2, self.dropout)
		conv_5_3 = conv_bn_relu(drop_5_2, [3, 3, 512, 512], 'C_5_3')
		pool_5 = max_pool_2x2(conv_5_3)

		# size of input is now Nx512
		flattened = flatten(pool_5, [-1, 512])
		
		fc1 = fc_bn_relu(flattened, [512, 512], 'FC1')
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
		return tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

	def set_learning_rate(self, lr):
		if self._session_running:
			self._session.run(self._learning_rate.assign(lr))
		else:
			print('learning rate can only be assigned in a running session.')
	
	@property
	def learning_rate(self):
		if self._session_running:
			return self._session.run(self._learning_rate)
		else:
			print('learning rate can only be evaluated in session.')
			return None