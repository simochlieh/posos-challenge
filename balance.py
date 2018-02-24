from imblearn.over_sampling import SMOTE


# Extension of SMOTE class in order to integrate it into a Pipeline
# As of now it raises errors because only X is passed through a sklearn pipeline
# In this case, both X and Y are modified by the smote.
class Balance(SMOTE):

    def __init__(self):
        super(Balance, self).__init__(k_neighbors=2)
        self.y = None

    def fit(self, X, y=None):
        self.y = y
        # self.set_params('k_neighbors':min(5, min(Counter(self.y).values()))})
        super(Balance, self).fit(X, y=y)

    def transform(self, X):
        out = super(Balance, self).sample(X, y=self.y)
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        out = self.transform(X)
        return out