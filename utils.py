import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import pickle


def create_dir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_drug_names_path():
    return '%sdrug_names' % (get_results_path())


def get_results_path():
    return 'results/'


def get_corr_lemm_path(label):
    return '%scorr_lemm/%s/input_train' % (get_results_path(),
                                           label)


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


def extend_class(cls):
    return lambda f: (setattr(cls, f.__name__, f) or f)


@extend_class(GridSearchCV)
def write_results(self):
    timestamp = str(datetime.now()).split('.')[0].replace(':', '.').replace(' ', '_')
    os.mkdir('./results/%s' % timestamp)

    with open('./results/%s/info.txt' % timestamp, 'w+') as f:
        f.write('##############################################')
        f.write('\nBest accuracy: %f' % (self.best_score_))
        f.write('\nobtained with:\n' + str(self.best_params_))
        f.write('\n\nAmong a 3 fold CV test on those params:\n' + str(self.param_grid))

    with open('./results/%s/model.pkl' % timestamp, 'wb+') as f:
        pickle.dump(self.best_estimator_, f)
