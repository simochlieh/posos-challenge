# Main for experiments
# from myxgb import MyXGB
from embedding import Tokenizer
from tfidf import MyVectorizer
from pipe import MyPipeline
from collections import Counter
from balance import Balance
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from balance import MySmote
import argparse
import pandas
import numpy
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

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
    X, y = load_data()

    # can be ignored: used for smote
    counter = Counter(y)
    ratios = {k: int(counter[k] - (counter[k] - numpy.mean(list(counter.values()))) * 0.2) for k in counter.keys()}

    # Bring objects in
    toke = Tokenizer(n_clusters=1)
    vecto = TfidfVectorizer(max_df=1.0)
    smote = Balance(ratio=ratios)
    fexKBest = SelectKBest(chi2, k=1500)
    fexModel = SelectFromModel(LinearSVC(C=0.1, dual=False, penalty='l2'))
    pca = TruncatedSVD(n_components=500)

    clf = GradientBoostingClassifier(n_estimators=10, verbose=1, subsample=1.0, max_depth=3)
    xgb = XGBClassifier(n_estimators=30, max_depth=6, objective='multi:softmax', silent=True, n_jobs=3)

    # Stack it all in a pipeline
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=1)
    pipe = MyPipeline(steps=
                      [('toke', toke),
                       ('vecto', vecto),
                       ('smote', smote),
                       # ('pca', pca),
                       ('fex', fexKBest),
                       ('clf', xgb)
                       ],
                      memory=memory
                      )

    # Build a Grid parameters, then the GSCV object
    # One dict is one grid to go through.
    params_grid = [
        {
            'toke__n_clusters': [1],
            'toke__max_df': [0.3],
            'vecto__max_df': [1.0],
            # 'pca': [TruncatedSVD(n_components=500)],
            'clf__n_estimators': [30],
            # 'clf__max_depth': [2, 3],
            # 'fex': [None, SelectKBest(chi2, k=200), SelectKBest(chi2, k=500), SelectKBest(chi2, k=1500)],
            # 'fex__estimator__C': [1, 10, 20],
            # 'fex__threshold': ["mean"],
            # 'fex__k': [1500],
            'smote': [None]
        }
    ]

    GSCV = GridSearchCV(pipe, n_jobs=3, param_grid=params_grid, verbose=3)

    # Now fit
    GSCV.fit(X, y)
    GSCV.write_results()
    print(GSCV.best_score_)
    print(GSCV.best_params_)
    rmtree(cachedir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
