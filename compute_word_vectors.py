from collections import defaultdict

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

MODEL_PATH = './wiki.fr/wiki.fr.bin'
EMBEDDING_DIRPATH = './results/embedding/fast_text_embedding_top_100_tfidf_no_corr/'
STOP_WORDS_FILEPATH = './data/stopwords-fr.txt'
DRUG_REPLACEMENT = 'médicament'
COMPUTE_STOP_WORDS = False  # If False we  read them from the file above
STOP_WORDS_TFIDF_MAX_DF = 0.1  # this is the max_df parameter for the TFIDF used to compute the stop words


class FastTextEmbedding:
    def __init__(self, sentences, y, drug_names_set, model_path, drug_description_embedding=True,
                 stop_words=None, do_correction=False, verbose=False, max_sentence_len=50, corrected_sent_path=None):
        assert len(sentences) == len(y), "List of sentences and y have different lengths. len(sentences) = %d, " \
                                         "len(y) = %d" % (len(sentences), len(y))
        self.sentences = sentences
        self.tokenized_sentences = []
        self.y = deepcopy(y.values)
        self.model_path = model_path
        self.do_correction = do_correction
        self.drug_names_set = drug_names_set
        self.verbose = verbose
        self.stop_words = stop_words
        self.drug_description_embedding = drug_description_embedding
        self.word2weight = None
        self.max_sentence_len = max_sentence_len
        self.corrected_sent_path = corrected_sent_path

    def tokenize_sentences(self):
        for i, sentence in tqdm(enumerate(self.sentences), desc="Tokenizing and correcting words...",
                                total=len(self.sentences)):
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

                # Correcting words
                if self.do_correction and not params.FR_DICT.check(word):
                    suggestions = params.FR_DICT.suggest(word)
                    if suggestions:
                        word = suggestions[0]

                sentence_v.append(word)

            if i == 0:
                print('\n' + ' '.join(sentence_v))

            self.tokenized_sentences.append(sentence_v)

    def write_corrected_sentences(self):
        with open(self.corrected_sent_path, 'w', encoding=params.UTF_8) as out:
            for sentence in self.tokenized_sentences:
                out.writelines(' '.join(sentence) + '\n')

    def read_corrected_sentences(self):
        with open(self.corrected_sent_path, encoding=params.UTF_8) as f:
            for line in f:
                self.tokenized_sentences.append(line.strip().split(' '))

    def compute_tfidf_weights(self):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(self.tokenized_sentences)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf)
        for w, i in tfidf.vocabulary_.items():
            self.word2weight[w] = tfidf.idf_[i]

    def run(self, save_directory=None):
        """

        :param save_directory: filepath where we save the text embedding
        :type save_directory: str
        :return: A 3d-array matrix storing the text embedding.
            The matrix is of shape (nb_sentences, max_sentence_length, embedding_size)
        """
        if self.do_correction:
            if os.path.exists(self.corrected_sent_path):
                self.do_correction = False
                self.read_corrected_sentences()
            else:
                # tokenize sentences
                self.tokenize_sentences()
        else:
            self.tokenize_sentences()

        if self.do_correction and self.corrected_sent_path:
            utils.create_dir(self.corrected_sent_path)
            self.write_corrected_sentences()
        # Computing tfidf weights
        self.compute_tfidf_weights()
        if self.verbose:
            print("Loading FastText model...")
        model = FastText.load_model(MODEL_PATH)
        drug_embedding_path = utils.get_drug_embedding_path()

        sentences_list = []

        if self.drug_description_embedding:
            try:
                with open(drug_embedding_path, 'rb') as f:
                    drug_embeddings = pickle.load(f)
            except FileNotFoundError:
                self.drug_description_embedding = False
                print('Drugs will be embedded as "médicament".')

        max_sentence_length = 0
        count_drugs = 0

        for i, sentence in tqdm(enumerate(self.tokenized_sentences), desc='Embedding words for each sentence...',
                                disable=not self.verbose, total=len(self.tokenized_sentences)):
            sentence_embedding = []
            chosen_words = []
            sorted_words = sorted(sentence, key=lambda w: - self.word2weight[w])
            idf_threshold = self.word2weight[sorted_words[min(self.max_sentence_len - 1, len(sorted_words) - 1)]]
            found_drug = False
            for word in sentence:
                # Skipping words having a low idf
                if self.word2weight[word] < idf_threshold or word in self.stop_words:
                    continue

                # Dealing with the drug name
                if unidecode(word) in self.drug_names_set:
                    # TODO: try something more complex
                    if self.drug_description_embedding and drug_embeddings:
                        try:
                            emb_w = drug_embeddings[unidecode(word)]
                            sentence_embedding.append(emb_w)
                            found_drug = True
                            continue
                        except KeyError:
                            word = DRUG_REPLACEMENT
                    else:
                        word = DRUG_REPLACEMENT

                # Embedding
                sentence_embedding.append(model.get_word_vector(word))
                chosen_words.append(word)
            if i == 0:
                print(len(chosen_words), len(sentence_embedding), '\n' + ' '.join(chosen_words))
            if found_drug:
                count_drugs += 1
            if sentence_embedding:
                sentences_list.append(sentence_embedding)
                if len(sentence_embedding) > max_sentence_length:
                    print(len(sentence_embedding), self.y[i], sentence)
            else:
                print("Warning: Found an empty sentence embedding for <%s>. Ignoring." % sentence)
                del self.y[i]

            # Updating max_sentence_length
            sentence_length = len(sentence_embedding)
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
        # print(np.array(sentences_list).mean())
        # Padding sentence matrices with 0 vectors
        text_embedding = []
        padding_token = model.get_word_vector(FastText.EOS)  # np.zeros((utils.get_embedding_dim(),))
        for sentence_embedding in sentences_list:
            sentence_length = len(sentence_embedding)
            sentence_embedding.extend([padding_token] * (max_sentence_length - sentence_length))
            text_embedding.append(sentence_embedding)

        # Deleting list of sentences
        del sentences_list
        print("\nFound %d sentences with drug names" % count_drugs)
        embeddings = np.array(text_embedding)
        print("\nSaving text embedding of shape %s" % str(embeddings.shape))

        X_train, X_test, y_train, y_test = train_test_split(embeddings, self.y, test_size=0.2, random_state=42)

        if save_directory:
            np.save(utils.get_X_train_path(save_directory), X_train)
            np.save(utils.get_X_test_path(save_directory), X_test)
            np.save(utils.get_y_train_path(save_directory), y_train)
            np.save(utils.get_y_test_path(save_directory), y_test)

        return embeddings


def main():
    input_train = pnd.read_csv(params.INPUT_TRAIN_FILENAME, sep=';')
    # input_test = pnd.read_csv(params.INPUT_TEST_FILENAME, sep=';')
    y = pnd.read_csv(utils.get_labels_path(), sep=';')

    drug_names_path = utils.get_drug_names_path()
    drug_names_df = pnd.read_csv(drug_names_path)
    drug_names_set = set(drug_names_df[params.DRUG_NAME_COL])

    stop_words = utils.compute_stop_words(input_train.question, max_df=STOP_WORDS_TFIDF_MAX_DF) if COMPUTE_STOP_WORDS \
        else utils.get_stop_words(STOP_WORDS_FILEPATH)
    if COMPUTE_STOP_WORDS:
        print("stop words: %s" % ', '.join(stop_words))

    fast_text_embedding = FastTextEmbedding(input_train.question, y.intention, drug_description_embedding=False,
                                            drug_names_set=drug_names_set, stop_words=[],
                                            model_path=MODEL_PATH, do_correction=False, verbose=True,
                                            corrected_sent_path='./results/corr/input_train', max_sentence_len=100)
    utils.create_dir(EMBEDDING_DIRPATH)
    fast_text_embedding.run(save_directory=EMBEDDING_DIRPATH)


if __name__ == '__main__':
    main()
