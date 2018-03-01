from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from embedding import Tokenizer
from tfidf import MyVectorizer
import argparse
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pnd
from tempfile import mkdtemp
from sklearn.decomposition import TruncatedSVD, PCA
import time


import utils
import params

parser = argparse.ArgumentParser(description="This is a test script to run sklearn pipelines on the data.")

parser.add_argument('-l',
                    '--label',
                    default='final',
                    help="This is used in many filepaths in order to distinguish each run.")


def main(args):
    start = time.time()
    input_train = pnd.read_csv(utils.get_corr_lemm_path(args.label))
    input_test = pnd.read_csv(utils.get_corr_lemm_path(args.label, test=True))
    y = pnd.read_csv(utils.get_labels_path(), sep=';')[params.LABELS_COL].values
    X_train, X_val, y_train, y_val = train_test_split(
        input_train, y, test_size=0.2, random_state=42)
    tokenizer = Tokenizer()
    vectorizer = MyVectorizer(is_sparse=True)
    pca = TruncatedSVD()
    svm = SVC(random_state=42)
    # Caching operations to avoid repetitions
    cachedir = mkdtemp()
    pipe = Pipeline([
        ('tokenizer', tokenizer),
        ('vectorizer', vectorizer),
        # ('pca', pca),
        ('svm', svm)
    ],
        memory=cachedir)

    params_grid = dict(
        tokenizer__do_clustering=[True],
        tokenizer__n_clusters=[2, 3, 4, 5],
        tokenizer__max_df=[1.],
        vectorizer__max_df=[1.],
        vectorizer__max_features=[None],
        svm__C=[40.],
        svm__gamma=[0.05],
        svm__kernel=['rbf'],
        # pca__n_components=[1000, 2000, 3000, 4000],
    )
    params_pipe = dict(
        tokenizer__do_clustering=True,
        tokenizer__n_clusters=2,  # , 5, 20],
        tokenizer__max_df=1.,  # , 0.2],
        vectorizer__max_df=1.,  # , 0.05],
        vectorizer__max_features=None,
        svm__C=40.,  # , 100., 10., 1.],
        svm__gamma=0.05,  # , 0.05, 0.1],
        svm__kernel='rbf',
        # pca__n_components=3000,
    )
    pipe.set_params(**params_pipe)
    pipe.fit(input_train, y)
    print(pipe.score(X_val, y_val))
    predictions = pipe.predict(input_test)
    utils.to_csv(predictions, './results/%s/svm/y_pred.csv' % datetime.strftime(datetime.now(), "%Y_%m_%dT%H_%M_%S"))

    # grid_search = GridSearchCV(pipe, n_jobs=3, cv=3, param_grid=params_grid, verbose=3)
    # grid_search.fit(X_train, y_train)
    # utils.write_results(grid_search)
    print("It took %.3f" % (time.time() - start))


if __name__ == '__main__':
    args_ = parser.parse_args()
    print(args_)
    main(args_)
