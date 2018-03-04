import pandas as pnd
import re
from fastText import FastText
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from unidecode import unidecode
import numpy as np
import pickle

import params
import utils

MODEL_PATH = './wiki.fr/wiki.fr.bin'
STOP_WORDS_FILEPATH = './data/stopwords-fr.txt'
DRUG_REPLACEMENT = 'médicament'
COMPUTE_STOP_WORDS = False  # otherwise we  read them from the file above
STOP_WORDS_TFIDF_MAX_DF = 0.1  # this is the max_df parameter for the TFIDF used to compute the stop words


class FastTextEmbedding:
    def __init__(self, sentences, y, drug_names_set, model_path, stop_words=None, do_correction=False, verbose=False):
        assert len(sentences) == len(y), "List of sentences and y have different lengths. len(sentences) = %d, " \
                                         "len(y) = %d" % (len(sentences), len(y))
        self.sentences = sentences
        self.y = y
        self.model_path = model_path
        self.do_correction = do_correction
        self.drug_names_set = drug_names_set
        self.verbose = verbose
        self.stop_words = stop_words

    def run(self, save_directory=None):
        """

        :param save_directory: filepath where we save the text embedding
        :type save_directory: str
        :return: A 3d-array matrix storing the text embedding.
            The matrix is of shape (nb_sentences, max_sentence_length, embedding_size)
        """
        if self.verbose:
            print("Loading FastText model...")
        # model = FastText.load_model(MODEL_PATH)

        sentences_list = []

        try:
            with open(utils.get_embedding_dirpath(), 'rb') as f:
                drug_embeddings = pickle.load(f)
        except FileNotFoundError:
                print('Drugs will be embedded as "médicament".')

        for sentence in tqdm(self.sentences, desc='Embedding words for each sentence...', disable=not self.verbose):
            sentence_embedding = []
            sentence = sentence.lower()
            splits = FastText.tokenize(sentence)
            for word in splits:
                # Skipping non-words
                if not re.match('(\w)+', word) or word in self.stop_words:
                    continue

                # Getting rid of the apostrophe and taking the following word
                apos_split = word.split("'")
                if len(apos_split) == 2:
                    _, word = apos_split
                    if not word:
                        continue

                # Dealing with the drug name
                if unidecode(word) in self.drug_names_set:
                    # TODO: try something more complex
                    try:
                        emb_w = drug_embeddings[word]
                        sentence_embedding.append(emb_w)
                        continue
                    except KeyError:
                        word = DRUG_REPLACEMENT

                # Correcting words
                if self.do_correction and not params.FR_DICT.check(word):
                    suggestions = params.FR_DICT.suggest(word)
                    if suggestions:
                        word = suggestions[0]

                # Embedding
                try:
                    sentence_embedding.append(model.get_word_vector(word))
                except NameError:
                    print('Uncomment line 44, model unloaded yet (5gig RAM required).')

            sentences_list.append(sentence_embedding)

        #     # Updating max_sentence_length
        #     sentence_length = len(sentence_embedding)
        #     if sentence_length > max_sentence_length:
        #         max_sentence_length = sentence_length
        #
        # # Padding sentence matrices with 0 vectors
        # text_embedding = []
        # for sentence_embedding in sentences_list:
        #     sentence_length = len(sentence_embedding)
        #     sentence_embedding.extend([np.zeros((utils.get_embedding_dim(),))] \
        #                               * (max_sentence_length - sentence_length))
        #     text_embedding.append(sentence_embedding)
        #
        # # Deleting list of sentences
        # del sentences_list

        embeddings = np.array(sentences_list)
        print("\nSaving text embedding of shape %s" % str(embeddings.shape))

        X_train, X_test, y_train, y_test = train_test_split(embeddings, self.y, test_size=0.2, random_state=42)

        if save_directory:
            np.save(utils.get_X_train_path(save_directory), X_train)
            np.save(utils.get_X_test_path(save_directory), X_test)
            np.save(utils.get_y_train_path(save_directory), y_train)
            np.save(utils.get_y_test_path(save_directory), y_test)

        return embeddings


if __name__ == '__main__':
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

    fast_text_embedding = FastTextEmbedding(input_train.question, y.intention,
                                            drug_names_set=drug_names_set, stop_words=stop_words,
                                            model_path=MODEL_PATH, do_correction=True, verbose=True)
    utils.create_dir(utils.get_embedding_dirpath())
    fast_text_embedding.run(save_directory=utils.get_embedding_dirpath())
