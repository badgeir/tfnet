
import numpy as np
import tensorflow as tf

import cifar_reader
import neural_network
from neural_network.dataset_handler import Dataset

def run():	
	#read untared cifar dataset from folder ./dataset and preprocess images and labels
	X, Y = cifar_reader.read_and_preprocess()
	
	dataset = Dataset(X, Y)
	del X, Y

	# build convolutional neural network
	network = neural_network.CifarNet()
	# start tf session and initialize tf variables
	network.start_session()
	
	n_epochs = 10
	dataset.set_batch_size(128)

	for epoch in range(n_epochs):		
		# test accuracy on validation batch
		x_batch, y_batch = dataset.validation_batch(512)
		print(network.accuracy(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.}))

		if epoch%4==3:
			lr = network.learning_rate
			print('setting learning rate to %f.'%(lr/10))
			network.set_learning_rate(lr/10)

		epoch_done = False
		while not epoch_done:
			# next batch
			epoch_done, x_batch, y_batch = dataset.next_training_batch()
			# add image noise
			x_batch = x_batch + np.random.normal(x_batch)*0.001
			# update weights
			network.train_batch(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 0.5})

	network.end_session()


if __name__=='__main__':
	run()