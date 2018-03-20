import math
import numpy
from utils import get_embedding_dim, get_parsing_dim
import time
from keras.utils import to_categorical


def batch_generator(input_data, y, batch_size, max_sent_length, n_channels=1, length_bounds=None):
    X = map(lambda s: numpy.stack(s[:max_sent_length]), input_data)

    # Faster filtering here than filtering on the fly in the while loop.
    if length_bounds is not None:
        Xyl = filter(lambda t: min(length_bounds) <= len(t[0]) <= max(length_bounds), zip(X, y, range(len(y))))
        X, y, steps = tuple(map(list, zip(*Xyl)))
        steps_per_epoch = math.ceil(len(steps) / batch_size)
    else:
        X = list(X)
        steps_per_epoch = math.ceil(len(X) / batch_size)

    """
    Yields embedded sentences in matrices of shape (max_sent_length, embedding_size, 1)
    """
    i = 0

    # Loops indefinitely, as precised in https://keras.io/models/sequential/
    while True:
        sl = slice(i * batch_size, (i + 1) * batch_size)

        mats = X[sl]
        # max_sent_length = numpy.max([m.shape[0] for m in mats])
        y_batch = y[sl]

        # pad all the matrices (sentences) one by one.
        e = get_embedding_dim() + get_parsing_dim()
        for k, m in enumerate(mats):
            mats[k] = numpy.vstack((mats[k], numpy.zeros(((max_sent_length - m.shape[0]), e))))
            if n_channels > 0:
                mats[k] = mats[k].reshape(max_sent_length, e, n_channels)

        # Now stack em all in a 3D tensor of shape (batch_size, sent_length, embedding_size)
        batch = numpy.array(mats)

        # Avoid storing too large numbers by modulo.

        i = (i + 1) % steps_per_epoch

        # Reshape y
        y_keras = to_categorical(y_batch, num_classes=51)

        yield (batch, y_keras)


def main():
    loaded = numpy.load('./results/embedding/numpy_saved_sparse.npy', mmap_mode='r')

    start = time.time()
    for i in range(100):
        a = loaded[(i % (8028 // 50)) * 50:((i + 1) % (8028 // 50)) * 50, :100, :] * 1
    end = time.time()
    print('mmap loaded access time %f' % ((end - start) / 100))

    b = batch_generator()
    a = next(b)
    print('starting chrono…')
    start = time.time()
    for i in range(100):
        a = next(b)
    print('stopping chrono…')
    end = time.time()
    print('generator access time %f' % ((end - start) / 100))
