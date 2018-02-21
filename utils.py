import os


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
