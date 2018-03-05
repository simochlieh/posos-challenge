import numpy
from utils import get_embedding_dim, get_labels_path, get_max_sent_length
import pandas
import time
from keras.utils import to_categorical
from params import BATCH_SIZE, TRAIN_STEPS_PER_EPOCH

EMBEDDING_FILEPATH = './results/embedding/fast_text_embedding_wo_stop_words/'


def batch_generator(input_data, y, batch_size, max_sent_length):
    X = list(map(lambda s: numpy.stack(s[:max_sent_length]), input_data))

    """
    Yields embedded sentences in matrices of shape (max_sent_length, embedding_size, 1)
    """
    i = 0

    # Loops indefinitely, as precised in https://keras.io/models/sequential/
    while True:
        bound = i % TRAIN_STEPS_PER_EPOCH
        sl = slice(bound, bound + batch_size)

        mats = X[sl]
        y_batch = y[sl]

        # pad all the matrices (sentences) one by one.
        e = get_embedding_dim()
        for k, m in enumerate(mats):
            mats[k] = numpy.vstack((mats[k], numpy.zeros(((max_sent_length - m.shape[0]), e)))) \
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
