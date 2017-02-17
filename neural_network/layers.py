import tensorflow as tf

def _weight_variable(shape, name, initial=None):
	if initial is None:
		initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def _bias_variable(shape, name, initial=None):
	if initial is None:
		initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def _conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1],
						  strides=[1,2,2,1], padding='SAME')

# convolution + batch normalization + relu layer
def conv_bn_relu(x, w_shape, name):
	W = _weight_variable(w_shape, 'W_%s'%name)
	b = _bias_variable([w_shape[3]], 'bias_%s'%name)
	conv = _conv2d(x, W) + b
	mean, var = tf.nn.moments(conv, [0])
	return tf.nn.relu( tf.nn.batch_normalization(conv, mean, var, 0, 1, 1e-6) )

def flatten(x, shape):
	return tf.reshape(x, shape)

# fully connected + batch normalization + relu layer
def fc_bn_relu(x, w_shape, name):
	W = _weight_variable(w_shape, 'W_%s'%name)
	b = _bias_variable([w_shape[1]], 'bias_%s'%name)
	z = tf.matmul(x, W) + b
	mean, var = tf.nn.moments(z, [0])
	return tf.nn.relu( tf.nn.batch_normalization(z, mean, var, 0, 1, 1e-6) )

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

#Peters special sauce
def defusion(x, x_size, name):
	initial = tf.truncated_normal([x_size, x_size], stddev=0.1)
	W = tf.Variable(initial, 'W_%s'%name)
	zero_diag = tf.ones([x_size, x_size], dtype=tf.float32) - tf.eye(x_size, dtype=tf.float32)
	W_zero_diag = tf.mul(W, zero_diag)
	b = bias_variable([x_size], 'bias_%s'%name)
	return tf.nn.tanh(tf.matmul(x, W_zero_diag) + b)