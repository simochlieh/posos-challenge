# This is the tf-idf script for the preprocessing step.
# it should be tested in the queue of the cleansing step.
# The read_file func is pointless as it wont be used in real framework.
# The sklearn tfidfvectorizer class was subclassed in order to integrate
# The whole process into a sklearn pipeline (very convenient for CV grid testing)
# This object stores the parameters it was fed with, even if those parameters are accessible
# Through the params.py file.

# TF-file for preprocessing step
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pnd
from copy import deepcopy
from scipy import sparse
import os
import json

import params
import utils

parser = argparse.ArgumentParser(description='This script takes as an input the corrected and stemmed data '
                                             'after adding drug information, and it outputs a vectorized form of '
                                             'the input sentences.')

parser.add_argument('--max-features',
                    type=int,
                    default=None,
                    help='This is the maximum number of features (vocabulary words) for the Tf-idf vectorizer')

parser.add_argument('--max-df',
                    type=float,
                    default=0.1,
                    help='This is the maximum document frequency considered. '
                         'Any word having a higher frequency is not taken into account')

parser.add_argument('--label',
                    default='final',
                    help="this parameter is used in the input and output file path.")


class MyVectorizer(TfidfVectorizer):
    # In practice,
    # all the sklearn parameters will be fed into this __init__ from the params.py file.

    def __init__(self, is_sparse=True, max_df=0.1, max_features=None, verbose=True):
        # Init mother
        self.verbose = verbose
        self.sparse = is_sparse
        super(MyVectorizer, self).__init__(max_df=max_df, max_features=max_features)

    def fit(self, sentences, y=None, **kwargs):
        if self.verbose:
            print('Fitting…')
        super(MyVectorizer, self).fit(sentences)

    def transform(self, sentences, **kwargs):
        if self.verbose:
            print('Transforming…')
        out = super(MyVectorizer, self).transform(sentences)
        # TODO: investigate KeyError exceptions
        try:
            if not self.sparse:
                out = out.toarray()
        except KeyError:
            pass
        return out

    def fit_transform(self, sentences, y=None, **kwargs):
        if self.verbose > 0:
            print('Fitting and Transforming...')
        out = super(MyVectorizer, self).fit_transform(sentences)
        try:
            if not self.verbose:
                out = out.toarray()
                return out
        except KeyError:
            return out

    # I don't think this will be useful
    # def _set_parameters(self):
    #     self.parameters = {
    #         'is_sparse': self.sparse,
    #         'max_df': self.max_df,
    #         'max_features': self.max_features,
    #         'verbose': self.verbose
    #     }


def read_lines(filepath):
    lines = []
    with open(filepath, encoding=params.UTF_8) as f:
        for line in f:
            lines.append(line.strip())

    return lines


def main(args):
    questions = read_lines(utils.get_tokenized_drugs_path(args.label))

    m = MyVectorizer(**vars(args))
    vectorized = m.fit_transform(questions)

    params_path = utils.get_vectorizer_params_path(args.label)
    utils.create_dir(params_path)
    with open(params_path, 'w') as out:
        # skipping keys that are not json serializable
        json.dump(m.parameters, out, skipkeys=True)

    stop_words_path = utils.get_stop_words_path(args.label)
    utils.create_dir(stop_words_path)
    with open(stop_words_path, 'w', encoding=params.UTF_8) as out:
        for stop_words in m.stop_words_:
            out.write(stop_words + '\n')

    output_path = utils.get_vectorized_data_path(args.label)
    if not os.path.exists(output_path):
        utils.create_dir(output_path)
    sparse.save_npz(output_path, vectorized)
