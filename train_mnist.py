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


network = MnistClassifier()
n_epochs = 10
network.start_session()

network.set_learning_rate(1)
for epoch in range(n_epochs):
    print('Epoch: {0}'.format(epoch))
    for x_batch, y_batch in dataset.batch_one_epoch(X_train, Y_train,
                                                batch_size=128):
        x_batch += np.random.normal(x_batch)*0.001
        # batch update weights
        batch_loss = network.train_batch(feed_dict={
                                                    network.x: x_batch,
                                                    network.y_: y_batch,
                                                    network.dropout: 0.7})
    predictions = network.predict(feed_dict={network.x: X_test,
                                         network.dropout: 1.})
    preds = np.argmax(predictions, axis=1)
    n_correct = np.where(preds == y)[0].size

    accuracy = n_correct / 10000
    print(accuracy)

    pred = network.predict(feed_dict={network.x: X_val,
                                      network.y_: Y_val,
                                      network.dropout: 0.})

network.save('mnist_final')
network.end_session()
