# from tfnet.datasets import rotten_reviews
import tfnet.dataset_handler as dt
import numpy as np
import tensorflow as tf

# X_train, X_val, Y_train, Y_val = rotten_reviews.load()

emb_sz = 5
X = np.random.randn(1000, 100, emb_sz, 1)
Y = np.mean(X, axis=1)
Y = np.mean(Y, axis=1)
Y = (Y > 2).astype(int)

x = tf.placeholder(tf.float32, (None, 100, 5, 1))
y = tf.placeholder(tf.float32, (None, 1))

filt_sz = 6
num_filt = 32
fs = [filt_sz, emb_sz, 1, num_filt]
W = tf.Variable(tf.truncated_normal(fs, stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[num_filt]), name="b")
conv = tf.nn.conv2d(
    x,
    W,
    strides=[1, 1, 1, 1],
    padding="VALID",
    name="conv")
flat = tf.contrib.layers.flatten(conv)
output = tf.contrib.layers.fully_connected(flat, 1)

loss = tf.reduce_mean(tf.abs(y - output))
optimizer = tf.train.AdadeltaOptimizer(0.1)\
            .minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for xb, yb in dt.batch_one_epoch(X, Y, batch_size=32):
            _, l = sess.run([optimizer, loss], feed_dict={x: xb, y: yb})
            print(l)