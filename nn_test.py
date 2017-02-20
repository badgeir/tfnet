
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
	
	n_epochs = 100
	
	train_loss_log = []

	network.start_session()
	network.set_learning_rate(0.001)
	for epoch in range(n_epochs):		
		# test accuracy on validation batch
		x_batch, y_batch = dataset.validation_batch(1024)
		val_loss, val_acc = network.validate_batch(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
		
		if epoch > 0:
			print('train acc/loss: %f/%f, val acc/loss: %f/%f'%(train_acc, train_loss_log[-1], val_acc, val_loss))

		if len(train_loss_log) > 1:
			# reduce learning rate if training is stagnating
			if train_loss_log[-1] > train_loss_log[-2]:
				lr = network.learning_rate
				print('setting learning rate from %f to %f.'%(lr, lr/10.))
				network.set_learning_rate(lr/10.)
				network.save(epoch)

		accu_loss, accu_acc, n_batches = 0., 0., 0
		for x_batch, y_batch in dataset.batch_until_epoch(batch_size=128):
			x_batch += np.random.normal(x_batch)*0.001
			# batch update weights
			batch_loss, batch_acc = network.train_batch(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 0.5})
			accu_loss += batch_loss
			accu_acc += batch_acc
			n_batches += 1

		train_loss_log.append(accu_loss/n_batches)
		train_acc = accu_acc / n_batches

	network.end_session()

if __name__=='__main__':
	run()