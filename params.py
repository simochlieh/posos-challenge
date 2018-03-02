# Parameters file for experiments
# Allows easier modifications and access to the parameters:
# Once the main is written, only this file needs to be modified in order to
# Proceed to various experiments

from enchant import Dict
import numpy as np

#################################################
# TF-IDF parameters
#################################################
"""
While the TruncatedSVD transformer works with any (sparse) feature matrix, 
using it on tf–idf matrices is recommended over raw frequency counts in an 
LSA/document processing setting. In particular, sublinear scaling and inverse 
document frequency should be turned on (sublinear_tf=True, use_idf=True) to 
bring the feature values closer to a Gaussian distribution, compensating for 
LSA’s erroneous assumptions about textual data.
"""

params_tfidf = {
    'input': "content",
    'encoding': "utf-8",
    'decode_error': "strict",
    'strip_accents': None,
    'lowercase': True,
    'preprocessor': None,
    'tokenizer': None,
    'analyzer': "word",
    'stop_words': None,
    'token_pattern': r"(?u)\b\w\w+\b",
    'ngram_range': (1, 1),
    'max_df': 1.0,
    'min_df': 1,
    'max_features': None,
    'vocabulary': None,
    'binary': False,
    'norm': "l2",
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': True,
    'sparse': True,
    'verbose': 0
}

#################################################
# Cleaning and lemmatization parameters
#################################################
FR_DICT = Dict("fr_FR")
FR_DICT_BLACKLIST = ('aspirine', 'carlin', 'morphine')
DRUG_NAMES_BLACKLIST = ('\n', 'anti', 'santé')
RCP_ENCODING = 'ISO-8859-1'
UTF_8 = 'utf-8'
DRUG_NAME_COL = 'name'
DRUG_COMPLETE_NAME_COL = 'complete_name'
DRUG_ID_COL = 'id'
INPUT_TRAIN_FILENAME = './data/input_train.csv'
INPUT_TEST_FILENAME = './data/input_test.csv'

RCP_FILENAME = './data/rcp/CIS.txt'
SENTENCE_ID = 'id'
RAW_SENTENCE_COL = 'raw_sentence'
CORR_LEMM_SENTENCE_COL = 'corr_lemm_sentence'
DRUG_NAMES_COL = 'drug_names'
DRUG_POS_COL = 'drug_ids'
FORMAT_DATE = '%Y_%m_%dT%H_%M_%S'
NB_LINES_DRUGS = 30148
LABELS_COL = 'intention'

#################################################
# Keras parameters
#################################################
BATCH_SIZE = 50
STEPS_PER_EPOCH = 8028 // BATCH_SIZE
CLASSES = list(range(51))

# CAn not set multithreading with batch generator.
keras_fit_params = {'epochs': 1,
                    'steps_per_epoch': STEPS_PER_EPOCH,
                    'verbose': 2,
                    'use_multiprocessing': False,
                    'workers': 1}
