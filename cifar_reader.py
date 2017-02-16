import numpy as np

def unpickle(filename):
	import pickle
	fo = open(filename, 'rb')
	dict = pickle.load(fo, encoding='latin1')
	fo.close()
	return dict

def read_cifar():
	data1 = unpickle('dataset/data_batch_1')
	data2 = unpickle('dataset/data_batch_2')
	data3 = unpickle('dataset/data_batch_3')
	data4 = unpickle('dataset/data_batch_4')
	data5 = unpickle('dataset/data_batch_5')

	# reshape and rescale images
	X1 = data1['data']
	X1 = X1.reshape((-1, 32, 32, 3), order='F').transpose(0, 2, 1, 3).astype(np.float32)/255.

	X2 = data2['data']
	X2 = X2.reshape((-1, 32, 32, 3), order='F').transpose(0, 2, 1, 3).astype(np.float32)/255.

	X3 = data3['data']
	X3 = X3.reshape((-1, 32, 32, 3), order='F').transpose(0, 2, 1, 3).astype(np.float32)/255.

	X4 = data4['data']
	X4 = X4.reshape((-1, 32, 32, 3), order='F').transpose(0, 2, 1, 3).astype(np.float32)/255.

	X5 = data5['data']
	X5 = X5.reshape((-1, 32, 32, 3), order='F').transpose(0, 2, 1, 3).astype(np.float32)/255.

	# concatenate all sets into one matrix
	X = np.concatenate((X1, X2, X3, X4, X5), axis=0)

	# clear som ram
	del X1, X2, X3, X4, X5

	# class labels
	y1 = np.array(data1['labels'])
	y2 = np.array(data2['labels'])
	y3 = np.array(data3['labels'])
	y4 = np.array(data4['labels'])
	y5 = np.array(data5['labels'])

	y = np.concatenate((y1, y2, y3, y4, y5), axis=0)
	
	return X, y

def preprocess_dataset(X, y):
	# mean_X = X.mean()
	r_mean_X = X[:,:,:,0].mean()
	g_mean_X = X[:,:,:,1].mean()
	b_mean_X = X[:,:,:,2].mean()
	
	#std_X = X.std()
	r_std_X = X[:,:,:,0].std()
	g_std_X = X[:,:,:,1].std()
	b_std_X = X[:,:,:,2].std()
	
	#X = (X-mean_X)/std_X
	X[:,:,:,0] = (X[:,:,:,0] - r_mean_X)/r_std_X
	X[:,:,:,1] = (X[:,:,:,1] - g_mean_X)/g_std_X
	X[:,:,:,2] = (X[:,:,:,2] - b_mean_X)/b_std_X
	
	# one-hot encode labels
	N_images = X.shape[0]
	Y = np.zeros((N_images, 10))
	Y[np.arange(N_images), y] = 1
	
	# double dataset size by flipping left-right
	X2 = np.fliplr(X)
	X = np.concatenate((X, X2),axis=0)
	Y = np.tile(Y, [2,1])

	return X, Y

def read_and_preprocess():
	X, Y = read_cifar()
	return preprocess_dataset(X, Y)
