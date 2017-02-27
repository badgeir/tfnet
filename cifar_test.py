import cifar_reader
from tfnet.models import CifarNet
import tfnet.dataset_handler as dataset


def calculate_accuracy(network, X, Y):
    correct_predictions, total = 0, 0
    for x_batch, y_batch in dataset.batch_one_epoch(X, Y, batch_size=100):
        total += 100
        correct_predictions += network.correct_predictions(feed_dict={
                                                           network.x: x_batch,
                                                           network.y_: y_batch,
                                                           network.dropout: 1.})
    val_accuracy = float(correct_predictions) / total
    return val_accuracy


def run():
    network = CifarNet()
    network.start_session()
    network.load_parameters('saved_models/cifarnet_final.ckpt')

    X_test, Y_test = cifar_reader.read_and_preprocess('dataset', 'data_batch_5')
    test_accuracy = calculate_accuracy(network, X_test, Y_test)
    print('\n\ntest accuracy: %f\n\n' % test_accuracy)

    network.end_session()


if __name__ == '__main__':
    run()
