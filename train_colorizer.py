from collections import deque
import numpy as np

import cifar_reader
from tfnet.models import ColorNet
import tfnet.dataset_handler as dataset

from matplotlib import pyplot as plt

def run():
    # read untared cifar dataset from folder ./dataset
    # and preprocess images and labels:
    X, Y = cifar_reader.read_and_preprocess('data_batch_1',
                                            'data_batch_2',
                                            'data_batch_3',
                                            'data_batch_4',
                                            dir='dataset')

    X_train, X_val, Y_train, Y_val = dataset.split(X, Y, test_size=0.2)
    del X, Y, Y_train, Y_val

    Y_train = X_train.copy()
    Y_val = X_val.copy()

    X_train = X_train.mean(axis=3)[:, :, :, None]
    X_val = X_val.mean(axis=3)[:, :, :, None]

    #fig = plt.figure()
    #fig.add_subplot(2, 2, 1)
    #plt.imshow(X_train[0])
    #fig.add_subplot(2, 2, 2)
    #plt.imshow(X_train[1])
    #fig.add_subplot(2, 2, 3)
    #plt.imshow(Y_train[0])
    #fig.add_subplot(2, 2, 4)
    #plt.imshow(Y_train[1])
    #plt.show()

    network = ColorNet()
    n_epochs = 100

    network.start_session()

    # start training
    for epoch in range(n_epochs):
        print('Epoch %d:' % epoch)
        accu_loss, accu_acc, n_batches = 0, 0, 0
        for x_batch, y_batch in dataset.batch_one_epoch(X_train, Y_train,
                                                        batch_size=128):
            # batch update weights
            batch_loss = network.train_batch(feed_dict={
                                                        network.x: x_batch,
                                                        network.y_: y_batch,
                                                        network.dropout: 0.7})
            accu_loss += batch_loss
            n_batches += 1
            if n_batches % 40 == 0:
                print('training loss: ', float(accu_loss)/n_batches)
                x_batch, y_batch = dataset.random_batch(X_val, Y_val, batch_size=128)
                val_loss = network.loss(feed_dict={network.x: x_batch,
                                                   network.y_: y_batch,
                                                   network.dropout: 1.})
                print('validation loss: ', val_loss)
        x_batch, y_batch = dataset.random_batch(X_train, Y_train, batch_size=2)
        outputs = network.output(feed_dict={network.x: x_batch, network.dropout: 1.})

        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(x_batch[0].squeeze(), cmap='gray')
        fig.add_subplot(2, 2, 2)
        plt.imshow(x_batch[1].squeeze(), cmap='gray')
        fig.add_subplot(2, 2, 3)
        plt.imshow(outputs[0])
        fig.add_subplot(2, 2, 4)
        plt.imshow(outputs[1])
        plt.show()

        x_batch, y_batch = dataset.random_batch(X_val, Y_val, batch_size=4)
        outputs = network.output(feed_dict={network.x: x_batch, network.dropout: 1.})

        fig = plt.figure()
        fig.add_subplot(2, 4, 1)
        plt.imshow(x_batch[0].squeeze(), cmap='gray')
        fig.add_subplot(2, 4, 2)
        plt.imshow(x_batch[1].squeeze(), cmap='gray')
        fig.add_subplot(2, 4, 3)
        plt.imshow(x_batch[2].squeeze(), cmap='gray')
        fig.add_subplot(2, 4, 4)
        plt.imshow(x_batch[3].squeeze(), cmap='gray')
        fig.add_subplot(2, 4, 5)
        plt.imshow(outputs[0])
        fig.add_subplot(2, 4, 6)
        plt.imshow(outputs[1])
        fig.add_subplot(2, 4, 7)
        plt.imshow(outputs[2])
        fig.add_subplot(2, 4, 8)
        plt.imshow(outputs[3])
        plt.show()

if __name__ == '__main__':
    run()