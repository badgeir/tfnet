import numpy as np
import os

def unpickle(filename):
	import pickle
	fo = open(filename, 'rb')
	dict = pickle.load(fo, encoding='latin1')
	fo.close()
	return dict

def read_cifar(dir, filenames):
	data = []
	for file in filenames:
		file_path = os.path.join(dir, file)
		data.append(unpickle(file_path))

	im_data = ()
	label_data = ()
	for x in data:
		images = x['data']
		images = images.reshape((-1, 32, 32, 3), order='F').transpose(0, 2, 1, 3).astype(np.float32)/255.
		im_data = im_data + (images, )

		label_data = label_data + (np.array(x['labels']), )
	X = np.concatenate(im_data, axis=0)
	y = np.concatenate(label_data, axis=0)
	del images, im_data, label_data, data
	
	return X, y

def preprocess_dataset(X, y):
	## mean_X = X.mean()
	#r_mean_X = X[:,:,:,0].mean()
	#g_mean_X = X[:,:,:,1].mean()
	#b_mean_X = X[:,:,:,2].mean()
	
	##std_X = X.std()
	#r_std_X = X[:,:,:,0].std()
	#g_std_X = X[:,:,:,1].std()
	#b_std_X = X[:,:,:,2].std()
	
	##X = (X-mean_X)/std_X
	#X[:,:,:,0] = (X[:,:,:,0] - r_mean_X)/r_std_X
	#X[:,:,:,1] = (X[:,:,:,1] - g_mean_X)/g_std_X
	#X[:,:,:,2] = (X[:,:,:,2] - b_mean_X)/b_std_X
	
	# one-hot encode labels
	N_images = X.shape[0]
	Y = np.zeros((N_images, 10))
	Y[np.arange(N_images), y] = 1
	
	# double dataset size by flipping left-right
	X2 = np.fliplr(X)
	X = np.concatenate((X, X2),axis=0)
	Y = np.tile(Y, [2,1])

	return X, Y

def read_and_preprocess(dir, *filenames):
	X, Y = read_cifar(dir, filenames)
	return preprocess_dataset(X, Y)
