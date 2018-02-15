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
    def __init__(self, **kwargs):
        # Get params
        self.parameters = deepcopy(params.params_tfidf)
        self.parameters.update(kwargs)
        keys = list(self.parameters.keys())

        for key in keys:
            if key not in params.params_tfidf:
                self.parameters.pop(key, None)
        # Init mother
        super(MyVectorizer, self).__init__(**self.parameters)

    def fit(self, sentences, **kwargs):
        print('Fitting…')
        super(MyVectorizer, self).fit(sentences)

    def transform(self, sentences, **kwargs):
        print('Transforming…')
        return super(MyVectorizer, self).transform(sentences)

    def fit_transform(self, sentences, **kwargs):
        print('Fitting and Transforming...')
        return super(MyVectorizer, self).fit_transform(sentences)

    def get_params(self, deep=True):
        return self.parameters


if __name__ == '__main__':
    args = parser.parse_args()

    questions = pnd.read_csv(utils.get_corr_lemm_path(args.label))[params.CORR_LEMM_SENTENCE_COL]

    m = MyVectorizer(**vars(args))
    vectorized = m.fit_transform(questions)

    print(vectorized.shape)
    print(m.get_params())
    print(m.stop_words_)

    output_path = utils.get_vectorized_data_path(args.label)
    if not os.path.exists(output_path):
        utils.create_dir(output_path)
    sparse.save_npz(output_path, vectorized)
