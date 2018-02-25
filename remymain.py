# Main for experiments

from embedding import Tokenizer
from tfidf import MyVectorizer
from pipe import MyPipeline

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from balance import Balance
import argparse
import pandas

from utils import *

parser = argparse.ArgumentParser(description="This script runs the main experiments.")

parser.add_argument('--n_clusters',
                    default=20,
                    help="This parameter is used to choose the number of clusters in which to classify drugs.")

parser.add_argument('--max_df',
                    default=0.3,
                    help="This parameter is used to choose the maximum frequency of words in "
                         "the drug tokenization process.")


def main(_args):
    # read data in
    input_train = pandas.read_csv(
        './results/corr_lemm/final/input_train')
    y_train = pandas.read_csv('/Users/remydubois/Desktop/posos/y_train.csv', sep=';').intention.values

    # Bring objects in
    toke = Tokenizer()
    vecto = TfidfVectorizer()
    smote = Balance()
    # pca = TruncatedSVD(n_components=100)
    clf = GradientBoostingClassifier()

    # Stack it all in a pipeline
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    pipe = MyPipeline(steps=
                    [('toke', toke),
                     ('vecto', vecto),
                     ('smote', smote),
                     # ('pca', pca),
                     ('clf', clf)
                     ],
                    memory=memory
                    )

    # Build a Grid parameters, then the GSCV object
    # One dict is one grid to go through.
    params_grid = [
        {
            # 'toke__n_clusters': [1, 5, 20],
            # 'toke__max_df': [0.3, 0.1],
            # 'vecto__max_df': [0.3, 0.1],
            # 'pca__n_components': [100],
            'clf__n_estimators': [100],
            'smote': [smote, None]
        }
    ]

    GSCV = GridSearchCV(pipe, n_jobs=3, param_grid=params_grid, verbose=3)

    # Now fit
    GSCV.fit(input_train, y_train)
    GSCV.write_results()
    rmtree(cachedir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
