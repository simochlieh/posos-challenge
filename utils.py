import os
from datetime import datetime
import pickle
import pandas
import pandas as pnd
import params
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np


def create_dir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_drug_names_path():
    return '%sdrug_names' % (get_results_path())


def get_results_path():
    return 'results/'


def get_corr_lemm_path(label, test=False):
    return '%scorr_lemm/%s/input_%s' % (get_results_path(),
                                        label, 'train' if not test else 'test')


def get_drug_embedding_path():
    return '%sembedding/drug_embeddings.pkl' % get_results_path()


def binary_search(array, element):
    """
    Not used yet
    :param array:
    :param element:
    :return:
    """
    if len(array) == 0:
        return False

    mid = len(array) // 2
    if element == array[mid]:
        return True
    elif element > array[mid] and mid + 1 < len(array):
        return binary_search(array[mid + 1:], element)
    elif element < array[mid]:
        return binary_search(array[:mid], element)

    return False


def string_to_bool(string):
    if string.lower() == 'true':
        return True
    else:
        return False


def get_vectorized_data_path(label):
    return '%svectorized/%s/input_train' % (get_results_path(), label)


def get_stop_words_path(label):
    return '%svectorized/%s/stop_words' % (get_results_path(), label)


def get_vectorizer_params_path(label):
    return '%svectorized/%s/params' % (get_results_path(), label)


def get_labels_path():
    return './data/y_train.csv'


def get_tokenized_drugs_path(label):
    return '%sdrug_tokenizer/%s/input_train' % (get_results_path(), label)


def get_embedding_dim():
    return 300


def extend_class(cls):
    def wrapper(f):
        return setattr(cls, f.__name__, f) or f

    return wrapper


@extend_class(GridSearchCV)
def write_results(self):
    timestamp = str(datetime.now()).split('.')[0].replace(':', '.').replace(' ', '_')
    os.mkdir('./results/%s' % timestamp)

    with open('./results/%s/info.txt' % timestamp, 'w+') as f:
        f.write('##############################################')
        f.write('\nBest accuracy: %f' % self.best_score_)
        f.write('\nobtained with:\n' + str(self.best_params_))
        f.write('\n\nAmong a 3 fold CV test on those params:\n' + str(self.param_grid))
        f.write('\n\nWhole CV results in the pickle object.')

    with open('./results/%s/model.pkl' % timestamp, 'wb+') as f:
        pickle.dump(self, f)


def load_data():
    input_train = pandas.read_csv(get_corr_lemm_path('final'))
    y_train = pandas.read_csv('/Users/remydubois/Desktop/posos/y_train.csv', sep=';').intention.values

    return input_train, y_train


def to_csv(predictions, filepath):
    create_dir(filepath)
    input_test = pnd.read_csv(params.INPUT_TEST_FILENAME, sep=';')
    df = pnd.DataFrame(columns=['ID', 'intention'])
    df['ID'] = input_test['ID']
    print(df)
    df = df.set_index('ID')
    df['intention'] = predictions
    df.to_csv(filepath)


def get_stop_words(filepath):
    stop_words = []
    with open(filepath, encoding=params.UTF_8) as f:
        for line in f:
            stop_words.append(line.strip())
    return stop_words


def compute_stop_words(sentences, max_df):
    tfidf = TfidfVectorizer(max_df=max_df)
    tfidf.fit(sentences)
    return tfidf.stop_words_


def get_X_train_path(dir):
    return os.path.join(dir, 'X_train.npy')


def get_y_train_path(dir):
    return os.path.join(dir, 'y_train.npy')


def get_X_test_path(dir):
    return os.path.join(dir, 'X_test.npy')


def get_y_test_path(dir):
    return os.path.join(dir, 'y_test.npy')


def get_shape(npy_file):
    npy = np.load(npy_file)
    return npy.shape
