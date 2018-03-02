import numpy
from utils import get_embedding_dim, get_labels_path
import pandas
import time
from keras.utils import to_categorical


def batch_generator(input_train, y, batch_size=50, max_sent_length=100):
    """
    Yields embedded sentences in matrices of shape (max_sent_length, embedding_size, 1)
    """
    X = list(map(lambda s: numpy.stack(s[:max_sent_length]), input_train))
    # # Check validity
    # if batch_size > len(input_train):
    #     raise ValueError('Batch size is too big (>dataset).')

    i = 0

    # Loops indefinitely, as precised in https://keras.io/models/sequential/
    while True:
        bound = i
        sl = slice(bound, bound + batch_size)

        mats = X[sl]
        y_batch = y[sl]

        # pad all the matrices (sentences) one by one.
        e = get_embedding_dim()
        for k, m in enumerate(mats):
            mats[k] = numpy.vstack((mats[k], numpy.zeros(((max_sent_length - m.shape[0]), e))))\
                .reshape(max_sent_length, e, 1)

        # Now stack em all in a 3D tensor of shape (batch_size, sent_length, embedding_size)
        batch = numpy.array(mats)

        # Avoid storing too large numbers by modulo.
        i = (i + batch_size) % len(X)

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
