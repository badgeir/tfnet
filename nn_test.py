
import numpy as np
import tensorflow as tf

import cifar_reader
from neural_network.models import CifarNet
from neural_network.dataset_handler import Dataset

def run():	
	#read untared cifar dataset from folder ./dataset and preprocess images and labels
	X, Y = cifar_reader.read_and_preprocess()

	dataset = Dataset(X, Y, test_size=0.1, val_size=0.2)
	del X, Y

	# build convolutional neural network
	network = CifarNet()
	
	n_epochs = 10
	dataset.set_batch_size(128)

	network.start_session()
	network.set_learning_rate(0.01)
	for epoch in range(n_epochs):		
		# test accuracy on validation batch
		x_batch, y_batch = dataset.validation_batch(1024)
		val_acc = network.accuracy(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
		val_loss = network.loss(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
		# test accuracy on validation training
		x_batch, y_batch = dataset.validation_batch(1024)
		train_acc = network.accuracy(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
		train_loss = network.loss(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
		
		print('train acc/loss: %f/%f, val acc/loss: %f/%f'%(train_acc, train_loss, val_acc, val_loss))

		if epoch%4==3:
			lr = network.learning_rate
			print('setting learning rate from %f to %f.'%(lr, lr/10.))
			network.set_learning_rate(lr/10.)

		epoch_done = False
		while not epoch_done:
			# next batch
			epoch_done, x_batch, y_batch = dataset.next_training_batch()
			# add image noise
			x_batch += np.random.normal(x_batch)*0.001
			# update weights
			network.train_batch(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 0.5})
	network.end_session()

if __name__=='__main__':
	run()