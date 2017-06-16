import sys
import zipfile
from tfnet.dataset_handler import split


def load():
    with zipfile.ZipFile('train.tsv.zip', 'r') as f:
        f.extractall()
    with open('train.tsv', 'r') as f:
        train = f.readlines()

    ids = []
    x = []
    y = []

    for line in train[1:]:
        pid, sid, phrase, sentiment = line.split('\t')
        if sid not in ids:
            ids.append(sid)
            x.append(phrase)
            y.append(int(sentiment))
    return split(x, y)


if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = load()
    assert(len(X_train) == 6823), 'Wrong number of samples in X_train: {}'.format(len(X_train))
    assert(len(Y_train) == 6823), 'Wrong number of samples in Y_train: {}'.format(len(Y_train))
    assert(len(X_val) == 1706), 'Wrong number of samples in X_val: {}'.format(len(X_val))
    assert(len(Y_val) == 1706), 'Wrong number of samples in Y_val: {}'.format(len(Y_val))