import numpy
from utils import get_embedding_dim, get_labels_path
from compute_word_vectors import EMBEDDING_FILEPATH
import pandas

def batch_generator(batch_size=50, max_sent_length=100):
    """
    Generates batches of sentences. Its main role is to avoid padding the whole
    dataset which would raise a tensor of dimensions (num_individuals = 8028,
    max_sentence_length_dataset = 626, embedding_size = 300). Instead, each batch is padded
    one by one which helps reducing memory charge.

    Returns a tuple (sentences, targets)
    """

    # First croppe the dataset in order to make it a round number of batches:
    l = numpy.load(EMBEDDING_FILEPATH)
    X = list(map(lambda s: numpy.stack(s[:max_sent_length]), l))
    Y = pandas.read_csv(get_labels_path(), sep=';').intention.values

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
        iterator = map(lambda m: numpy.vstack((m, numpy.zeros((max_sent_length - m.shape[0], e)))), mats)

        # Now stack em all in a 3D tensor of shape (batch_size, sent_length, embedding_size)
        batch = numpy.stack(iterator)

        # Avoid storing too large numbers by modulo.
        i = (i + batch_size) % len(X)

        yield (batch, y)
