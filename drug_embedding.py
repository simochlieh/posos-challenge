from collections import defaultdict
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from fastText import FastText
import pandas as pnd

import params
import utils

MODEL_PATH = './wiki.fr/wiki.fr.bin'


class TfidfEmbeddingVectorizer(object):
    def __init__(self, fasttext_model, embedding_dim):
        self.fasttext_model = fasttext_model
        self.word2weight = None
        self.dim = embedding_dim

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf)
        for w, i in tfidf.vocabulary_.items():
            self.word2weight[w] = tfidf.idf_[i]

    def transform(self, X):
        return np.array([
                np.mean([self.fasttext_model.get_word_vector(w) * self.word2weight[w] for w in words] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


if __name__ == '__main__':
    descriptions_df = pnd.read_csv(utils.get_drugs_indication_path(), encoding=params.UTF_8)
    fasttext_model = FastText.load_model(MODEL_PATH)
    tfidf_emb = TfidfEmbeddingVectorizer(fasttext_model, utils.get_embedding_dim())
    tfidf_emb.fit(descriptions_df.descriptions)
    embeddings = tfidf_emb.transform(descriptions_df.descriptions)

    embeddings_dict = {}
    for i in range(descriptions_df.shape[0]):
        embeddings_dict[descriptions_df.loc[i, 'drug_names']] = embeddings[i]

    with open(utils.get_drug_embedding_path(), 'wb') as f:
        pickle.dump(embeddings_dict, f)
