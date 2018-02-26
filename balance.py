from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from pipe import MyPipeline
import numpy


# Extension of SMOTE class in order to integrate it into a Pipeline
# As of now it raises errors because only X is passed through a sklearn pipeline
# In this case, both X and Y are modified by the smote.

# As of 25/02, it still needs some work as the n_neighbors arguments messes around.
# the set_params does not seem to work that much.
# Works fine otherwise.
class MySmote(SMOTE):

    def __init__(self, k_neighbors=3, n_jobs=1, ratio='all'):
        super(MySmote, self).__init__(k_neighbors=k_neighbors, n_jobs=n_jobs, ratio=ratio)
        self.y = None

    def fit(self, X, y=None):
        # needs to store y because it can not be passed as arg to transform
        # according the sklearn pipe design, but it is necessary in our case.
        self.y = y
        counter = Counter(y)
        # self.set_params(**{'k_neighbors': min(5, min(counter.values()))})
        if type(self.ratio) == int:
            self.set_params(**{'ratio': {k: max(self.ratio, counter[k]) for k in counter.keys()}})

        super(MySmote, self).fit(X, y=y)

    def transform(self, X):
        out = super(MySmote, self).sample(X, y=self.y)
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        out = self.transform(X)
        return out


class MyRus(RandomUnderSampler):

    def __init__(self, ratio='all'):
        super(MyRus, self).__init__(ratio=ratio)
        self.y = None

    def fit(self, X, y=None):
        self.y = y

        counter = Counter(y)

        if type(self.ratio) == int:
            self.set_params(**{'ratio': {k: min(self.ratio, counter[k]) for k in counter.keys()}})

        super(MyRus, self).fit(X, y=y)

    def transform(self, X):
        return super(MyRus, self).sample(X, y=self.y)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


class Balance(MyPipeline):

    def __init__(self, ratio=50):
        self.ratio = ratio
        super(Balance, self).__init__(steps=[('smote', MySmote(ratio=ratio)), ('rus', MyRus(ratio=ratio))])


def reduce_gaps(val):
    def inter(iterable):
        counter = Counter(iterable)
        ratios = {k: int(counter[k] - (counter[k] - numpy.mean(list(counter.values()))) * val) for k in counter.keys()}
        return ratios

    return inter
