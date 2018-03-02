import numpy
from utils import get_embedding_dim, get_labels_path, get_max_sent_length
from compute_word_vectors import EMBEDDING_FILEPATH
import pandas
import time
from keras.utils import to_categorical
from params import BATCH_SIZE


def batch_generator(batch_size=BATCH_SIZE, max_sent_length=get_max_sent_length()):
    # """This function is just a wrapper for batcher(), which actually generates the batches.
    # This wrapper is useful for loading the X and Y datasets before calling yield (otherwise,
    # X and Y would be loaded at the first call to yield which would cause the first batch to be
    # slower).
    # This wrapper can be safely removed with no other impact than increasing first batch loading time.
    # """
    # First croppe the dataset in order to make it a round number of batches:
    l = numpy.load(EMBEDDING_FILEPATH)
    X = list(map(lambda s: numpy.stack(s[:max_sent_length]), l))
    Y = pandas.read_csv(get_labels_path(), sep=';').intention.values

    #def batcher():
    """
    Generates batches of sentences. Its main role is to avoid padding the whole
    dataset which would raise a tensor of dimensions (num_individuals = 8028,
    max_sentence_length_dataset = 626, embedding_size = 300). Instead, each batch is padded
    one by one which helps reducing memory charge.

    Returns a tuple (sentences, targets)

    Loading time of a batch through this generator: ~13ms
    Loading time of a batch through numpy load with mmap: ~50ms
    """

    # Check validity
    if batch_size > len(l):
        raise ValueError('Batch size is too big (>dataset).')

    i = 0

    # Loops indefinitely, as precised in https://keras.io/models/sequential/
    while True:
        bound = i
        sl = slice(bound, bound + batch_size)

        mats = X[sl]
        y = Y[sl]

        # padd all the matrices (sentences) one by one.
        e = get_embedding_dim()
        iterator = map(
            lambda m: numpy.vstack((m, numpy.zeros((max_sent_length - m.shape[0], e)))), mats)

        reshaped = map(lambda m: m.reshape(max_sent_length, e, 1), iterator)

        # Now stack em all in a 3D tensor of shape (batch_size, sent_length, embedding_size)
        batch = numpy.stack(reshaped)

        # Avoid storing too large numbers by modulo.
        i = (i + batch_size) % len(X)

        # Reshape y
        y = to_categorical(y, num_classes=51)
        yield (batch, y)

    # return batcher()


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

