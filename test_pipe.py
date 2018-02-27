from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from embedding import Tokenizer
from tfidf import MyVectorizer
import argparse
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pnd
from tempfile import mkdtemp
from sklearn.decomposition import TruncatedSVD, PCA

import utils
import params

parser = argparse.ArgumentParser(description="This is a test script to run sklearn pipelines on the data.")

parser.add_argument('-l',
                    '--label',
                    default='final',
                    help="This is used in many filepaths in order to distinguish each run.")


def main(args):
    input_train = pnd.read_csv(utils.get_corr_lemm_path(args.label))
    y = pnd.read_csv(utils.get_labels_path(), sep=';')[params.LABELS_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        input_train, y, test_size=0.2, random_state=42)
    tokenizer = Tokenizer()
    vectorizer = MyVectorizer(is_sparse=False)
    pca = PCA()
    svm = SVC(random_state=42)
    # Caching operations to avoid repetitions
    cachedir = mkdtemp()
    pipe = Pipeline([
        ('tokenizer', tokenizer),
        ('vectorizer', vectorizer),
        ('pca', pca),
        ('svm', svm)
    ],
        memory=cachedir)

    params_grid = dict(
        tokenizer__n_clusters=[1],
        tokenizer__max_df=[0.1, 0.2],
        vectorizer__max_df=[0.1],
        vectorizer__max_features=[None],
        svm__C=[1., 50., 100.],
        svm__gamma=[0.1, 0.05, 0.01],
        svm__kernel=['rbf'],
        pca__n_components=[100, 1000, 6000],
    )
    params_pipe = dict(
        tokenizer__n_clusters=1,  # , 5, 20],
        tokenizer__max_df=0.1,  # , 0.2],
        vectorizer__max_df=0.1,  # , 0.05],
        vectorizer__max_features=4000,
        svm__C=50.,  # , 100., 10., 1.],
        svm__gamma=0.01,  # , 0.05, 0.1],
        svm__kernel='rbf',
    )
    # pipe.set_params(**params_pipe)
    # pipe.fit(X_train, y_train)
    # print(pipe.score(X_test, y_test))
    grid_search = GridSearchCV(pipe, n_jobs=1, cv=5, param_grid=params_grid, verbose=3)
    grid_search.fit(X_train, y_train)
    utils.write_results(grid_search)


if __name__ == '__main__':
    args_ = parser.parse_args()
    print(args_)
    main(args_)
