
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

import cifar_reader
import neural_network

def shuffle_dataset(X, Y):
	random_idx = np.arange(X.shape[0])
	np.random.shuffle(random_idx)

	X = X[random_idx]
	Y = Y[random_idx]

	return X, Y

def train_val_test_split(X, y):
		X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
		return X_train, y_train, X_val, y_val, X_test, y_test

def run():	
	#read untared cifar dataset from folder ./dataset and preprocess images and labels
	X, y = cifar_reader.read_and_preprocess()

	#split into training, validation and test sets
	X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)	
	
	#clear ram
	del X, y

	# build convolutional neural network
	network = neural_network.CifarNet()
	# start tf session and initialize tf variables
	network.start_session()
	
	n_epochs = 10
	batch_size = 100
	for epoch in range(n_epochs):
		X_train, y_train = shuffle_dataset(X_train, y_train)
		
		# test accuracy on validation batch
		random_idx = np.random.choice(np.arange(X_val.shape[0]), size=512, replace=False)
		x_batch = X_val[random_idx]
		y_batch = y_val[random_idx]
		print(network.accuracy(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.}))

		for i in range(int(X_train.shape[0]/batch_size)):
			# next batch
			x_batch = X_train[i*batch_size : i*batch_size+batch_size]
			# add image noise
			x_batch = x_batch + np.random.normal(x_batch)*0.001
			y_batch = y_train[i*batch_size : i*batch_size+batch_size]

			# update weights
			network.train_batch(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 0.5})

	network.end_session()


if __name__=='__main__':
	run()