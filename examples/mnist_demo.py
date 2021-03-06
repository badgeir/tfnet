import numpy as np
from tfnet.models import MnistClassifier
import tfnet.dataset_handler as dataset
import tfnet.datasets.mnist as mnist


X_train, Y_train, X_val, Y_val, X_test, Y_test = mnist.load()

network = MnistClassifier()

epochs = 10
network.start_session()
for epoch in range(epochs):
    for x_batch, y_batch in dataset.batch_one_epoch(X_train, Y_train,
                                                    batch_size=128):
        # batch update weights
        batch_loss = network.train_batch(feed_dict={
                                                    network.x: x_batch,
                                                    network.y_: y_batch,
                                                    network.dropout: 0.7})
    predictions = network.predict(feed_dict={network.x: X_val,
                                             network.dropout: 1.})
    preds = np.argmax(predictions, axis=1)
    labels = np.argmax(Y_val, axis=1)
    n_correct = np.where(preds == labels)[0].size
    accuracy = n_correct / X_val.shape[0]
    print('Epoch: {}, Validation score: {}'.format(epoch, accuracy))

predictions = network.predict(feed_dict={network.x: X_test,
                                         network.dropout: 1.})
preds = np.argmax(predictions, axis=1)
labels = np.argmax(Y_test, axis=1)
n_correct = np.where(preds == labels)[0].size
accuracy = n_correct / X_test.shape[0]
print('Final score on test data: {}'.format(accuracy))

network.save('mnist_final')
network.end_session()
