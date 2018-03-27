import csv

import pandas as pnd
import params
from fastText import train_unsupervised
import numpy as np
import os
from scipy import stats


# Because of fasttext we don't need to account for OOV
def compute_similarity(data_path):
    def similarity(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / n1 / n2

    mysim = []
    gold = []

    with open(data_path, 'rb') as fin:
        for line in fin:
            tline = line.split()
            word1 = tline[0].lower()
            word2 = tline[1].lower()

            v1 = model.get_word_vector(word1)
            v2 = model.get_word_vector(word2)
            d = similarity(v1, v2)
            mysim.append(d)
            gold.append(float(tline[2]))

    corr = stats.spearmanr(mysim, gold)
    dataset = os.path.basename(data_path)
    correlation = corr[0] * 100
    return dataset, correlation, 0


if __name__ == '__main__':
    train_data = pnd.read_csv(params.INPUT_TRAIN_FILENAME, sep=';')['question']
    valid_data = pnd.read_csv(params.INPUT_TEST_FILENAME, sep=';')['question']
    all_data = pnd.concat([train_data, valid_data])
    train_data_path = 'train_data.txt'
    valid_data_path = 'test_data.txt'
    print(type(train_data))
    train_data.to_frame().to_csv(train_data_path, header=False, sep=';', index=False, quoting=csv.QUOTE_NONE,
                                 escapechar=' ')
    valid_data.to_frame().to_csv(valid_data_path, header=False, sep=';', index=False, quoting=csv.QUOTE_NONE,
                                 escapechar=' ')

    model = train_unsupervised(
        input=train_data_path,
        model='skipgram',
        pretrainedVectors='./wiki.fr/wiki.fr.bi'
    )
    model.save_model("my_model.bin")
