
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
	network.load_parameters('saved_models/cifarnet.ckpt-1337')

	learning_rate = 0.00001
	network.set_learning_rate(learning_rate)

	# start training
	for epoch in range(n_epochs):		

		if epoch > 0 and epoch % 5 == 0:
			# test accuracy on validation batch
			correct_predictions, total = 0, 0
			for x_batch, y_batch in dataset.validation_epoch(batch_size=100):
				correct_predictions += network.correct_predictions(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
				total += 100
			val_accuracy = float(correct_predictions) / total
			print('\n\nvalidation accuracy: %f\n\n'%val_accuracy)

			print('train acc/loss: %f/%f, val acc/loss: %f/%f'%(train_acc, train_loss_log[-1], val_acc, val_loss))
			network.save(epoch)

		if len(train_loss_log) > 1:
			# reduce learning rate if training is plateauing
			if train_loss_log[-2] - train_loss_log[-1] < train_loss_log[-1]*learning_rate*40:
				print('setting learning rate from %f to %f.'%(learning_rate, learning_rate/10.))
				learning_rate = learning_rate/10.
				network.set_learning_rate(learning_rate)

		accu_loss, accu_acc, n_batches = 0., 0., 0
		for x_batch, y_batch in dataset.training_epoch(batch_size=128):
			x_batch += np.random.normal(x_batch)*0.001
			# batch update weights
			batch_loss, batch_acc = network.train_batch(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 0.7})
			accu_loss += batch_loss
			accu_acc += batch_acc
			n_batches += 1

		train_loss_log.append(accu_loss/n_batches)
		train_acc = accu_acc / n_batches
	
	# test accuracy
	correct_predictions, total = 0, 0
	for x_batch, y_batch in dataset.test_epoch(batch_size=100):
		correct_predictions += network.correct_predictions(feed_dict={network.x: x_batch, network.y_: y_batch, network.dropout: 1.})
		total += 100
	test_accuracy = float(correct_predictions) / total
	print('\n\ntest accuracy: %f\n\n'%test_accuracy)

	network.end_session()

if __name__=='__main__':
	run()