import pandas as pnd
import re
from fastText import FastText
from tqdm import tqdm
from unidecode import unidecode
import numpy as np


import params
import utils

MODEL_PATH = './wiki.fr/wiki.fr.bin'
EMBEDDING_FILEPATH = './results/embedding/fast_text_embedding.npy'
STOP_WORDS_FILEPATH = './data/stopwords-fr.txt'
DRUG_REPLACEMENT = 'médicament'
EMBEDDING_SIZE = 300
COMPUTE_STOP_WORDS = True  # otherwise we  read them from the file above
STOP_WORDS_TFIDF_MAX_DF = 0.1  # this is the max_df parameter for the TFIDF used to compute the stop words


class FastTextEmbedding:
    def __init__(self, sentences, drug_names_set, model_path, stop_words=None, do_correction=False, verbose=False):
        self.sentences = sentences
        self.model_path = model_path
        self.do_correction = do_correction
        self.drug_names_set = drug_names_set
        self.verbose = verbose
        self.stop_words = stop_words

    def run(self, save_path=None):
        """

        :param save_path: filepath where we save the text embedding
        :type save_path: str
        :return: A 3d-array matrix storing the text embedding.
            The matrix is of shape (nb_sentences, max_sentence_length, embedding_size)
        """
        if self.verbose:
            print("Loading FastText model...")
        model = FastText.load_model(MODEL_PATH)

        sentences_list = []
        max_sentence_length = 0

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
                if unidecode(word) in drug_names_set:
                    # TODO: try something more complex
                    word = DRUG_REPLACEMENT

                # Correcting words
                if self.do_correction and not params.FR_DICT.check(word):
                    suggestions = params.FR_DICT.suggest(word)
                    if suggestions:
                        word = suggestions[0]

                # Embedding
                sentence_embedding.append(model.get_word_vector(word))

            sentences_list.append(sentence_embedding)

            # Updating max_sentence_length
            sentence_length = len(sentence_embedding)
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
        #
        # # Padding sentence matrices with 0 vectors
        # text_embedding = []
        # for sentence_embedding in sentences_list:
        #     sentence_length = len(sentence_embedding)
        #     sentence_embedding.extend([np.zeros((EMBEDDING_SIZE,))] * (max_sentence_length - sentence_length))
        #     text_embedding.append(sentence_embedding)
        #
        # # Deleting list of sentences
        # del sentences_list

        embeddings = np.array(sentences_list)
        print("\nSaving text embedding of shape %s" % str(embeddings.shape))

        if save_path:
            np.save(save_path, embeddings)

        return embeddings


if __name__ == '__main__':
    input_train = pnd.read_csv(params.INPUT_TRAIN_FILENAME, sep=';')
    input_test = pnd.read_csv(params.INPUT_TEST_FILENAME, sep=';')
    input_data = pnd.concat([input_train, input_test])

    drug_names_path = utils.get_drug_names_path()
    drug_names_df = pnd.read_csv(drug_names_path)
    drug_names_set = set(drug_names_df[params.DRUG_NAME_COL])

    stop_words = utils.compute_stop_words(input_train.question, max_df=STOP_WORDS_TFIDF_MAX_DF) if COMPUTE_STOP_WORDS \
        else utils.get_stop_words(STOP_WORDS_FILEPATH)
    if COMPUTE_STOP_WORDS:
        print("stop words: %s" % ', '.join(stop_words))

    fast_text_embedding = FastTextEmbedding(input_train.question, drug_names_set=drug_names_set, stop_words=stop_words,
                                            model_path=MODEL_PATH, do_correction=True, verbose=True)
    utils.create_dir(EMBEDDING_FILEPATH)
    fast_text_embedding.run(save_path=EMBEDDING_FILEPATH)
