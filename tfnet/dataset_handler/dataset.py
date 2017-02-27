from sklearn.model_selection import train_test_split
import numpy as np


def split(X, Y, test_size=0.2, random_seed=42):
    return train_test_split(X, Y, test_size=test_size,
                            random_state=random_seed)


def batch_one_epoch(X, Y, batch_size=32):
    random_idx = np.arange(X.shape[0])
    np.random.shuffle(random_idx)

    current_idx = 0
    while current_idx + batch_size <= X.shape[0]:
        cur_randoms = random_idx[range(current_idx, current_idx + batch_size)]
        x_batch = X[cur_randoms]
        y_batch = Y[cur_randoms]
        yield x_batch, y_batch
        current_idx += batch_size


def random_batch(X, Y, batch_size=32):
    random_idx = np.random.choice(np.arange(X.shape[0]),
                                  batch_size, replace=False)
    return X[random_idx], Y[random_idx]
