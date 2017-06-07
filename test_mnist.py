import pickle
import gzip
import numpy as np
from matplotlib import pyplot as plt
from tfnet.models import MnistClassifier
import tfnet.dataset_handler as dataset


# Load the dataset
f = gzip.open(r'mnist\mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
_, _, test_set = u.load()
f.close()

X_test = test_set[0].reshape(10000, 28, 28, 1)
Y_test = np.zeros((10000, 10))
Y_test[np.arange(10000), test_set[1]] = 1
y = test_set[1]

network = MnistClassifier()
network.start_session()
network.load_parameters('saved_models/mnist_final.ckpt')

predictions = network.predict(feed_dict={network.x: X_test[0][None, :],
                                         network.dropout: 1.})
hundreds = (predictions*100).astype(int).astype(float)/100
print(list(hundreds))

# n_correct = np.where(preds == y)[0].size

# accuracy = n_correct / 10000
# print(accuracy)

# network.end_session()
