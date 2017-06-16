import gzip
import pickle
import numpy as np


def load():
    f = gzip.open(r'tfnet\datasets\mnist\mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()

    X_train = train_set[0].reshape(50000, 28, 28, 1)
    Y_train = np.zeros((50000, 10))
    Y_train[np.arange(50000), train_set[1]] = 1

    X_val = valid_set[0].reshape(10000, 28, 28, 1)
    Y_val = np.zeros((10000, 10))
    Y_val[np.arange(10000), valid_set[1]] = 1

    X_test = test_set[0].reshape(10000, 28, 28, 1)
    y = test_set[1]
    Y_test = np.zeros((10000, 10))
    Y_test[np.arange(10000), test_set[1]] = 1

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
