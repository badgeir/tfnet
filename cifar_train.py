from collections import deque
import numpy as np

import cifar_reader
from tfnet.models import CifarNet
import tfnet.dataset_handler as dataset


def calculate_accuracy(network, X, Y):
    correct_predictions, total = 0, 0
    for x_batch, y_batch in dataset.batch_one_epoch(X, Y, batch_size=100):
        correct_predictions += network.correct_predictions(feed_dict={
                                                           network.x: x_batch,
                                                           network.y_: y_batch,
                                                           network.dropout: 1.})
        total += 100
    val_accuracy = float(correct_predictions) / total
    return val_accuracy


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

    # build convolutional neural network
    network = CifarNet()
    n_epochs = 100
    # train_loss_log = deque(maxlen=3)
    val_acc_log = deque(maxlen=3)

    network.start_session()
    network.load_parameters('saved_models/cifarnet_final.ckpt')

    # learning_rate = 0.1
    # network.set_learning_rate(learning_rate)

    # start training
    for epoch in range(n_epochs):
        print('Epoch %d:' % epoch)
        if epoch > 0:
            # test accuracy on validation batch
            val_accuracy = calculate_accuracy(network, X_val, Y_val)
            val_acc_log.append(val_accuracy)
            print('validation accuracy: %f' % (val_accuracy))

            if len(val_acc_log) == 3:
                if val_acc_log[2] - val_acc_log[0] < 0.003:
                    # validation score plateauing, stop training
                    break

            network.save('cifarnet_train')

        accu_loss, accu_acc, n_batches = 0., 0., 0
        for x_batch, y_batch in dataset.batch_one_epoch(X_train, Y_train,
                                                        batch_size=128):
            x_batch += np.random.normal(x_batch)*0.001
            # batch update weights
            batch_loss, batch_acc = network.train_batch(feed_dict={
                                                        network.x: x_batch,
                                                        network.y_: y_batch,
                                                        network.dropout: 0.7})
            accu_loss += batch_loss
            accu_acc += batch_acc
            n_batches += 1

            if n_batches % 20 == 0:
                # add summaries
                x_batch, y_batch = dataset.random_batch(X_train, Y_train,
                                                        batch_size=256)
                network.training_summary(feed_dict={network.x: x_batch,
                                                    network.y_: y_batch,
                                                    network.dropout: 1.})
                x_batch, y_batch = dataset.random_batch(X_val, Y_val,
                                                        batch_size=256)
                network.validation_summary(feed_dict={network.x: x_batch,
                                                      network.y_: y_batch,
                                                      network.dropout: 1.})
                train_acc = accu_acc / n_batches
                train_loss = accu_loss / n_batches
                print('training loss / accuracy: %f / %f'
                      % (train_loss, train_acc))

    # test accuracy
    X_test, Y_test = cifar_reader.read_and_preprocess('dataset', 'data_batch_5')
    test_accuracy = calculate_accuracy(network, X_test, Y_test)
    print('\n\ntest accuracy: %f\n\n' % test_accuracy)

    network.save('cifarnet_final')
    network.end_session()


if __name__ == '__main__':
    run()
