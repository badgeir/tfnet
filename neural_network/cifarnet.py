from neural_network.base_network import NeuralNetwork
from neural_network.layers import max_pool_2x2, conv_bn_relu, flatten, fc_bn_relu, dropout

import tensorflow as tf

class CifarNet(NeuralNetwork):
	def __init__(self):
		NeuralNetwork.__init__(self)

	def network_definition(self):
		layer1 = conv_bn_relu(self.x, [7, 7, 3, 32], 'C1')
		layer2 = conv_bn_relu(layer1, [5, 5, 32, 32], 'C2')
		layer3 = max_pool_2x2(layer2)
		layer4 = conv_bn_relu(layer3, [5, 5, 32, 64], 'C3')
		layer5 = conv_bn_relu(layer4, [5, 5, 64, 64], 'C4')
		layer6 = max_pool_2x2(layer5)

		# size of input is now N*8x8x64
		layer7 = flatten(layer6, [-1, 8*8*64])
		
		layer8 = fc_bn_relu(layer7, [8*8*64, 512], 'FC1')
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		layer9 = dropout(layer8, self.dropout_keep_prob)

		layer10 = fc_bn_relu(layer9, [512, 10], 'Y')
		return layer10

	def loss_definition(self):
		return tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(self._network, self.y_))

	def optimizer_definition(self):
		return tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)
