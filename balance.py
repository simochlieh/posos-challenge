from imblearn.over_sampling import SMOTE
from collections import Counter


# Extension of SMOTE class in order to integrate it into a Pipeline
# As of now it raises errors because only X is passed through a sklearn pipeline
# In this case, both X and Y are modified by the smote.

# As of 25/02, it still needs some work as the n_neighbors arguments messes around.
# the set_params does not seem to work that much.
# Works fine otherwise.
class Balance(SMOTE):

    def __init__(self, k_neighbors=3, n_jobs=1):
        super(Balance, self).__init__(k_neighbors=k_neighbors, n_jobs=n_jobs)
        self.y = None

    def fit(self, X, y=None):
        # needs to store y because it can not be passed as arg to transform
        # according the sklearn pipe design, but it is necessary in our case.
        self.y = y
        counter = Counter(y)
        # self.set_params(**{'k_neighbors': min(5, min(counter.values()))})
        super(Balance, self).fit(X, y=y)

    def transform(self, X):
        out = super(Balance, self).sample(X, y=self.y)
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        out = self.transform(X)
        return out
