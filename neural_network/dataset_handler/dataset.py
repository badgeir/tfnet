from sklearn.model_selection import train_test_split
import numpy as np

def train_val_test_split(X, y, test_size=0.1, val_size=0.2):
		X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
		val_size = val_size/(1. - test_size)
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
		return X_train, y_train, X_val, y_val, X_test, y_test


class Dataset(object):
	def __init__(self, X, Y, test_size=0.1, val_size=0.2):
		# split dataset into train, test and validation sets
		self.X_train, self.Y_train, self.X_val, self.Y_val,\
			self.X_test, self.Y_test = train_val_test_split(X, Y, test_size=test_size, val_size=val_size)

	def _shuffle_training_data(self):
		random_idx = np.arange(self.X_train.shape[0])
		np.random.shuffle(random_idx)

		self.X_train = self.X_train[random_idx]
		self.Y_train = self.Y_train[random_idx]

	def batch_until_epoch(self, batch_size=32):
		self._shuffle_training_data()

		current_idx = 0
		while current_idx + batch_size <= self.X_train.shape[0]:
			x_batch = self.X_train[range(current_idx, current_idx + batch_size)]
			y_batch = self.Y_train[range(current_idx, current_idx + batch_size)]
			yield x_batch, y_batch
			current_idx += batch_size

	def training_epoch(self, batch_size=32):
		current_idx = 0
		while current_idx + batch_size <= self.X_test.shape[0]:
			x_batch = self.X_test[range(current_idx, current_idx + batch_size)]
			y_batch = self.Y_test[range(current_idx, current_idx + batch_size)]
			yield x_batch, y_batch
			current_idx += batch_size

	def training_batch(self, batch_size):
		random_idx = np.random.choice(np.arange(self.X_train.shape[0]), batch_size, replace=False)
		return self.X_train[random_idx], self.Y_train[random_idx]

	def validation_batch(self, batch_size):
		random_idx = np.random.choice(np.arange(self.X_val.shape[0]), batch_size, replace=False)
		return self.X_val[random_idx], self.Y_val[random_idx]

	def test_batch(self, batch_size):
		random_idx = np.random.choice(np.arange(self.X_test.shape[0]), batch_size, replace=False)
		return self.X_test[random_idx], self.Y_test[random_idx]