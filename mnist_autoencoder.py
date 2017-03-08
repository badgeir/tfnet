from collections import deque
import numpy as np

import cifar_reader
from tfnet.models import CifarAutoEncoder
import tfnet.dataset_handler as dataset

def run():
    # read untared cifar dataset from folder ./dataset
    # and preprocess images and labels:
    X, Y = cifar_reader.read_and_preprocess('data_batch_1',
                                            'data_batch_2',
                                            'data_batch_3',
                                            'data_batch_4',
                                            dir='dataset')

    X_train, X_val, Y_train, Y_val = dataset.split(X, Y, test_size=0.2)
    del X, Y

    network = CifarAutoEncoder()
    n_epochs = 100

    network.start_session()

    # start training
    for epoch in range(n_epochs):
        print('Epoch %d:' % epoch)
        accu_loss, accu_acc, n_batches = 0, 0, 0
        for x_batch, _ in dataset.batch_one_epoch(X_train, Y_train,
                                                        batch_size=128):
            # batch update weights
            batch_loss, batch_acc = network.train_batch(feed_dict={
                                                        network.x: x_batch,
                                                        network.y_: x_batch })
            accu_loss += batch_loss
            accu_acc += batch_acc
            n_batches += 1
            print(float(accu_loss)/n_batches)

if __name__ == '__main__':
    run()