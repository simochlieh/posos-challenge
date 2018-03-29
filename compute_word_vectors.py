from collections import defaultdict, Counter

import os
import pandas as pnd
import re
from fastText import FastText
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from unidecode import unidecode
import numpy as np
import pickle
from copy import deepcopy

import params
import utils

np.seterr(divide='raise', invalid='raise')

MODEL_PATH = './wiki.fr/wiki.fr.bin'
EMBEDDING_DIRPATH = './results/embedding/fast_text_embedding_top_50_tfidf_no_corr_w_drug_emb_test_0/'
STOP_WORDS_FILEPATH = './data/stopwords-fr.txt'
DRUG_REPLACEMENT = 'médicament'
COMPUTE_STOP_WORDS = False  # If False we  read them from the file above
STOP_WORDS_TFIDF_MAX_DF = 0.1  # this is the max_df parameter for the TFIDF used to compute the stop words


class FastTextEmbedding:
    def __init__(self, sentences, y, drug_names_set, model_path, test_input, drug_description_embedding=True,
                 stop_words=None, do_correction=False, verbose=False, max_sentence_len=50, corrected_sent_path=None,
                 y_val=None):

        self.sentences = sentences
        self.tokenized_sentences = []
        self.y_train = deepcopy(y.values)
        self.model_path = model_path
        self.do_correction = do_correction
        self.drug_names_set = drug_names_set
        self.verbose = verbose
        self.stop_words = stop_words
        self.drug_description_embedding = drug_description_embedding
        self.word2idf = None
        self.max_sentence_len = max_sentence_len
        self.corrected_sent_path = corrected_sent_path
        self.test_input = test_input
        self.tokenized_test_sent = []
        self.y_val = np.array(y_val) if y_val is not None else None

    def tokenize_sentences(self, X, test=False):
        tokenized_sentences = []
        labels = []

        for i, sentence in tqdm(enumerate(X), desc="Tokenizing and correcting words...",
                                total=len(X)):

            sentence = sentence.strip().lower()

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

                # Handling hyphens
                hyphen_split = word.split("-")
                if len(hyphen_split) > 1:
                    for sub_word in hyphen_split:
                        if sub_word:
                            sentence_v.append(sub_word)
                    continue

                # Correcting words
                if self.do_correction and not params.FR_DICT.check(word) and unidecode(word) not in self.drug_names_set:
                    continue

                sentence_v.append(word)

            if len(sentence_v) <= 1000 or test:
                tokenized_sentences.append(sentence_v)
                labels.append(self.y_train[i])
        return tokenized_sentences, labels

    @staticmethod
    def write_corrected_sentences(X, labels, path):
        with open(path, 'w', encoding=params.UTF_8) as out:
            for i, sentence in enumerate(X):
                out.writelines(' '.join(sentence) + ((';' + str(labels[i])) if labels else '') + '\n')

    def read_corrected_sentences(self):
        tokenized_sent = []
        labels = []
        with open(self.corrected_sent_path, encoding=params.UTF_8) as f:
            for line in f:
                split = line.strip().split(';')
                sent = split[0].strip()
                label = int(split[1].strip())
                tokenized_sent.append(sent.split(' '))
                labels.append(label)

        return tokenized_sent, labels

    def compute_tfidf_weights(self, X, labels):
        sent_per_class = {}
        for i, sentence in enumerate(X):
            if labels[i] in sent_per_class:
                sent_per_class[labels[i]] += sentence
            else:
                sent_per_class[labels[i]] = deepcopy(sentence)
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(sent_per_class.values())
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf
        max_idf = max(tfidf.idf_)

        self.word2idf = defaultdict(lambda: max_idf)
        for w, i in tfidf.vocabulary_.items():
            self.word2idf[w] = tfidf.idf_[i]

    def run(self, save_directory=None):
        """

        :param save_directory: filepath where we save the text embedding
        :type save_directory: str
        :return: A 3d-array matrix storing the text embedding.
            The matrix is of shape (nb_sentences, max_sentence_length, embedding_size)
        """
        # tokenize sentences
        self.tokenized_sentences, labels = self.tokenize_sentences(self.sentences)
        self.tokenized_test_sent, _ = self.tokenize_sentences(self.test_input, test=True)
        self.write_corrected_sentences(self.tokenized_sentences, labels, path=self.corrected_sent_path)
        # Computing tfidf weights
        self.compute_tfidf_weights(self.tokenized_sentences, labels)

        print("\nLoading FastText model...")
        model = FastText.load_model(MODEL_PATH)
        drug_embedding_path = utils.get_drug_embedding_path()
        drug_embeddings = None
        if self.drug_description_embedding:
            try:
                with open(drug_embedding_path, 'rb') as f:
                    drug_embeddings = pickle.load(f)
            except FileNotFoundError:
                self.drug_description_embedding = False
                print('\nDrugs will be embedded as "médicament".')

        embeddings, labels = self.embedding(self.tokenized_sentences, model=model, drug_embeddings=drug_embeddings,
                                            labels=labels)
        test_embeddings, _ = self.embedding(self.tokenized_test_sent, model=model, drug_embeddings=drug_embeddings,
                                            test=True)

        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, stratify=labels,
                                                            random_state=41)

        if save_directory:
            np.save(utils.get_X_train_path(save_directory), X_train)
            np.save(utils.get_X_test_path(save_directory), X_test)
            np.save(utils.get_y_train_path(save_directory), y_train)
            np.save(utils.get_y_test_path(save_directory), y_test)

            np.save(utils.get_test_embeddings_path(save_directory), test_embeddings)
            if self.y_val is not None:
                np.save(utils.get_y_test_embeddings_path(save_directory), self.y_val)
                print("y_val shape is %s" % str(self.y_val.shape))

        return embeddings

    def embedding(self, X, model, drug_embeddings, labels=None, test=False):

        sentences_list = []

        max_sentence_length = 0
        count_drugs_emb = 0
        count_drugs_match = 0
        all_processed_sent = []
        for i, sentence in tqdm(enumerate(X), desc='Embedding words for each sentence...',
                                disable=not self.verbose, total=len(X)):
            sentence_embedding = []
            chosen_words = []
            sorted_words = sorted(sentence, key=lambda w: - self.word2idf[w])
            try:
                idf_threshold = self.word2idf[sorted_words[min(self.max_sentence_len - 1, len(sorted_words) - 1)]]
            except IndexError:
                print("\nThe sentence id %d <%s> label %d became empty after processing." % (i, sentence,
                                                                                             labels[i] if labels
                                                                                             else -1))
            found_drug_emb = False
            found_drug_match = False
            count_words = 0
            for word in sentence:
                # Skipping words having a low idf
                if count_words >= self.max_sentence_len or self.word2idf[word] < idf_threshold \
                        or word in self.stop_words:
                    continue

                # Dealing with the drug name
                if unidecode(word) in self.drug_names_set:
                    found_drug_match = True
                    # TODO: try something more complex
                    if self.drug_description_embedding and drug_embeddings:
                        try:
                            emb_w = drug_embeddings[unidecode(word)]
                            sentence_embedding.append(emb_w / np.sqrt(emb_w.dot(emb_w)))
                            found_drug_emb = True
                            continue
                        except KeyError:
                            word = DRUG_REPLACEMENT
                    else:
                        word = DRUG_REPLACEMENT

                # Embedding
                word_vec = model.get_word_vector(word)
                sentence_embedding.append(word_vec / np.sqrt(word_vec.dot(word_vec)))
                chosen_words.append(word)
                count_words += 1
            all_processed_sent.append(chosen_words)
            if found_drug_emb:
                count_drugs_emb += 1
            if found_drug_match:
                count_drugs_match += 1
            sentences_list.append(sentence_embedding)
            if len(sentence_embedding) > self.max_sentence_len:
                print('\n', len(sentence_embedding), self.y_train[i] if not test else None, sentence)

            # Updating max_sentence_length
            sentence_length = len(sentence_embedding)
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
        # print(np.array(sentences_list).mean())
        # Padding sentence matrices with 0 vectors
        utils.create_dir('./results/preprocessed/X_%s' % ('train' if not test else 'test'))
        self.write_corrected_sentences(all_processed_sent, labels, path='./results/preprocessed/X_%s' %
                                                                        ('train' if not test else 'test'))
        text_embedding = []
        padding_token = np.zeros((utils.get_embedding_dim(),))  # model.get_word_vector(FastText.EOS)
        for sentence_embedding in sentences_list:
            sentence_length = len(sentence_embedding)
            sentence_embedding.extend([padding_token]  # / np.sqrt(padding_token.dot(padding_token))]
                                      * (max_sentence_length - sentence_length))
            text_embedding.append(sentence_embedding)

        # Deleting list of sentences
        del sentences_list
        print("\nEmbedded drug names in %d sentences" % count_drugs_emb)
        print("\nMatched drug names in %d sentences" % count_drugs_match)
        embeddings = np.array(text_embedding)
        print("\nSaving text embedding of shape %s" % str(embeddings.shape))

        if labels:
            # Removing classes with 1 element
            nb_per_class = Counter(labels)
            classes_to_remove = []
            for key, value in nb_per_class.items():
                if value == 1:
                    classes_to_remove.append(key)
            print(len(nb_per_class))
            print("\nClasses to remove %s" % classes_to_remove)

            remove_ops = 0
            for i, class_ in enumerate(labels):
                if class_ in classes_to_remove:
                    del labels[i]
                    embeddings = np.delete(embeddings, i, 0)
                    remove_ops += 1

            print("\nRemoved %d points" % remove_ops)
        return embeddings, labels


def main():
    input_train = pnd.read_csv(params.INPUT_TRAIN_FILENAME, sep=';')

    input_test = pnd.read_csv(params.INPUT_TEST_FILENAME, sep=';', encoding='utf-8-sig')
    y = pnd.read_csv(utils.get_labels_path(), sep=';')

    input_train_tr, input_train_v, y_tr, y_v = train_test_split(input_train.question, y.intention, test_size=0.2,
                                                                stratify=y.intention, random_state=41)

    drug_names_path = utils.get_drug_names_path()
    drug_names_df = pnd.read_csv(drug_names_path)
    drug_names_set = set(drug_names_df[params.DRUG_NAME_COL])

    stop_words = utils.compute_stop_words(input_train.question, max_df=STOP_WORDS_TFIDF_MAX_DF) if COMPUTE_STOP_WORDS \
        else utils.get_stop_words(STOP_WORDS_FILEPATH)
    if COMPUTE_STOP_WORDS:
        print("stop words: %s" % ', '.join(stop_words))

    fast_text_embedding = FastTextEmbedding(input_train.question, y.intention, drug_description_embedding=False,
                                            drug_names_set=drug_names_set, stop_words=[],
                                            model_path=MODEL_PATH, do_correction=True, verbose=True,
                                            corrected_sent_path='./results/corr/input_train', max_sentence_len=195,
                                            test_input=input_test.question)
    utils.create_dir(EMBEDDING_DIRPATH)
    fast_text_embedding.run(save_directory=EMBEDDING_DIRPATH)


if __name__ == '__main__':
    main()
