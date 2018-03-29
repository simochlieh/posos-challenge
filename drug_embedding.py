from collections import defaultdict
import numpy as np
import pickle

import re
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from fastText import FastText
import pandas as pnd
from tqdm import tqdm

import params
import utils

MODEL_PATH = './wiki.fr/wiki.fr.bin'


class TfidfEmbeddingVectorizer(object):
    def __init__(self, fasttext_model, embedding_dim, max_nb_words):
        self.fasttext_model = fasttext_model
        self.word2weight = None
        self.dim = embedding_dim
        self.max_nb_words = max_nb_words

    @staticmethod
    def tokenize_sentences(X):
        tokenized_sentences = []
        for i, sentence in tqdm(enumerate(X), desc="Tokenizing words...",
                                total=len(X)):
            sentence = sentence.lower()
            splits = word_tokenize(sentence, language='french')
            sentence_v = []
            for word in splits:
                if not re.match('(\w)+', word):
                    continue

                # Getting rid of the apostrophe and taking the following word
                apos_split = word.split("'")
                if len(apos_split) == 2:
                    _, word = apos_split
                    if not word:
                        continue

                sentence_v.append(word)

            if i == 1069:
                print('\n' + ' '.join(sentence_v))

            tokenized_sentences.append(sentence_v)
        return tokenized_sentences

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
        # return np.array([
        #         np.mean([self.fasttext_model.get_word_vector(w) * self.word2weight[w] for w in words], axis=0)
        #         for words in X
        #     ])
        embedding_matrix = []
        for i, words in enumerate(X):
            sentence_embedding = []
            list_words = []
            sorted_words = sorted(words, key=lambda w: - self.word2weight[w])

            if i == 1069:
                print([(w, self.word2weight[w]) for w in sorted_words])

            idf_threshold = self.word2weight[sorted_words[min(self.max_nb_words - 1, len(sorted_words) - 1)]]
            # total_weight = np.sum([self.word2weight[w] for w in sorted_words if self.word2weight[w] >= idf_threshold])
            if i == 1069:
                print(idf_threshold)  # / total_weight)
            for w in words:
                if w == 'nf':
                    w = 'médicament'
                if self.word2weight[w] < idf_threshold:
                    continue
                sentence_embedding.append(self.fasttext_model.get_word_vector(w) * self.word2weight[w])  # / total_weight)
                list_words.append(w)
            if i == 1069:
                print(len(list_words), ' '.join(list_words))
            embedding_matrix.append(np.mean(sentence_embedding, axis=0))
        return embedding_matrix


def main():
    descriptions_df = pnd.read_csv(utils.get_drugs_indication_path(), encoding=params.UTF_8)
    print("Loading FastText model...")
    fasttext_model = FastText.load_model(MODEL_PATH)
    tfidf_emb = TfidfEmbeddingVectorizer(fasttext_model, utils.get_embedding_dim(), max_nb_words=10)
    tokenized_sent = tfidf_emb.tokenize_sentences(descriptions_df.descriptions)
    tfidf_emb.fit(tokenized_sent)
    embeddings = tfidf_emb.transform(tokenized_sent)

    embeddings_dict = {}
    for i in range(descriptions_df.shape[0]):
        embeddings_dict[descriptions_df.loc[i, 'drug_names']] = embeddings[i]

    with open(utils.get_drug_embedding_path(), 'wb') as f:
        pickle.dump(embeddings_dict, f)


if __name__ == '__main__':
    main()
