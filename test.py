import sys
import numpy as np
import tfnet.dataset_handler as dh
try:
    X = np.random.randn(100, 3)
    Y = np.random.randn(100, 1)
    x, y = dh.random_batch(X, Y, batch_size=32)
    assert(x.shape == (32, 3)), 'Shape of x is wrong: {}'.format(x.shape)
    assert(y.shape == (32, 1)), 'Shape of y is wrong: {}'.format(y.shape)
    sys.exit(0)
except Exception as e:
    print(e)
    sys.exit(1)