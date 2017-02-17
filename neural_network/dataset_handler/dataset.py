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

		self._current_idx = 0

	def _shuffle():
		random_idx = np.arange(X.shape[0])
		np.random.shuffle(random_idx)

		self.X = self.X[random_idx]
		self.Y = self.Y[random_idx]

	def next_batch(self, batch_size):
		raise NotImplementedError