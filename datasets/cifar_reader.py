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
        images = images.reshape((-1, 32, 32, 3), order='F').\
            transpose(0, 2, 1, 3).astype(np.float32)/255.
        im_data = im_data + (images, )

        label_data = label_data + (np.array(x['labels']), )
    X = np.concatenate(im_data, axis=0)
    y = np.concatenate(label_data, axis=0)
    del images, im_data, label_data, data

    return X, y


def preprocess_dataset(X, y):

    # one-hot encode labels
    N_images = X.shape[0]
    Y = np.zeros((N_images, 10))
    Y[np.arange(N_images), y] = 1

    # double dataset size by flipping left-right
    X2 = np.flip(X, axis=2)
    X = np.concatenate((X, X2), axis=0)
    Y = np.tile(Y, [2, 1])

    return X, Y


def read_and_preprocess(*filenames, dir=''):
    X, Y = read_cifar(dir, filenames)
    return preprocess_dataset(X, Y)

