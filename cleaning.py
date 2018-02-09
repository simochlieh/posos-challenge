import argparse

import os
from enchant import Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from tqdm import tqdm
import pandas as pnd
import re
from unidecode import unidecode
from datetime import datetime

# notes: # regarder le set de médicaments extraits + enlever les trucs chelou
         # regarder qques phrases au hasard sans médicaments et voir pk?
FR_DICT = Dict("fr_FR")
FR_DICT_BLACKLIST = ('aspirine', 'carlin', 'morphine')
DRUG_NAMES_BLACKLIST = ('\n', 'anti', 'santé')
RCP_ENCODING = 'ISO-8859-1'
UTF_8 = 'utf-8'
DRUG_NAME_COL = 'name'
DRUG_COMPLETE_NAME_COL = 'complete_name'
DRUG_ID_COL = 'id'
INPU_DATA_FILENAME = './data/input_train.csv'
RCP_FILENAME = './data/rcp/CIS.txt'
SENTENCE_ID = 'id'
RAW_SENTENCE_COL = 'raw_sentence'
CORR_LEMM_SENTENCE_COL = 'corr_lemm_sentence'
DRUG_NAMES_COL = 'drug_names'
DRUG_IDS_COL = 'drug_ids'
FORMAT_DATE = '%Y_%m_%dT%H_%M_%S'
MAX_LINES = 9000
NB_DRUGS = 30148


parser = argparse.ArgumentParser(description='This description is shown when -h or --help are passed as arguments.')
parser.add_argument('--required_0',
                    type=int,
                    choices=[1, 2, 3],
                    required=False,
                    help='This is a required parameter')

parser.add_argument('--multi_integer',
                    type=int,
                    default=[1, 2],
                    nargs='+',
                    help='This is a multi integer parameter. You have to provide +(at least one) integers.')

parser.add_argument('-f',
                    '--foo',
                    action='store_true',
                    help='This is a flag parameter it can be set or not.')


def create_dir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_drug_names_path():
    return '%sdrug_names' % (get_results_path())


def get_results_path():
    return 'results/'


def get_corr_lemm_path(now=None):
    return '%scorr_lemm/%s/input_train' % (get_results_path(), datetime.strftime(now, FORMAT_DATE) if now else 'final')


def remove_blacklisted_words_from_dict(blacklist):
    """
    Some blacklisted french words
    (e.g. carline is a real word but also a drug name)
    :param blacklist: list of blacklisted words
    :return: Nothing
    """
    for word in blacklist:
        FR_DICT.remove(word)


def requirements():
    nltk.download('nonbreaking_prefixes')
    nltk.download('perluniprops')


def suggest_accented(word):
    """
    Tries to correct unaccented words.
    Only handles the accents on letter 'e', most common situation.
    :param word: word to correct
    :return: correction
    """
    word = word.lower()
    for i in range(len(word)):
        if word[i] == 'e':
            for replacement in ('é', 'è', 'ê'):
                new_word = word[:i] + replacement + word[i+1:]
                if FR_DICT.check(new_word):
                    return new_word
    return None


def extract_drug_names(filename, sep_regex, id_col_idx, name_col_idx):
    """
    Extract list of drug names
    :param sep_regex: separator regex string
    :param filename: string
    :param id_col_idx: int, column index of drug id
    :param name_col_idx: int, column index of drug name
    :return: Pandas Dataframe id: name
    """
    drug_names = []
    with open(filename, encoding='ISO-8859-1') as f:
        for line in tqdm(f, desc='pre-processing drug names', total=NB_DRUGS):
            splits = re.split(sep_regex, line)
            if len(splits) > 3 and splits[name_col_idx] not in DRUG_NAMES_BLACKLIST:
                names = re.split(r'\s|,|-|/|\\|_', splits[name_col_idx])
                drug_name = None
                for name in names:
                    if len(name) > 3 and re.match('([a-zA-Z])+', name) and not FR_DICT.check(name.lower()):
                        drug_name = name
                        break
                if drug_name:
                    suggestion = suggest_accented(drug_name)
                    if suggestion:
                        drug_name = suggestion
                    drug_names.append((int(splits[id_col_idx]), unidecode(drug_name.lower()), ' '.join(names).lower()))
    drug_names = sorted(list(set(drug_names)), key=lambda x: x[1])
    df = pnd.DataFrame(drug_names, columns=[DRUG_ID_COL, DRUG_NAME_COL, DRUG_COMPLETE_NAME_COL])
    df = df.set_index(DRUG_ID_COL)
    return df


def read_lines(filename, sep):
    """
    Read lines of input file
    :param sep: separator character
    :param filename: string
    :return: array of (id, sentence)
    """
    lines = []
    with open(filename, encoding=UTF_8) as f:
        f.readline()
        i = 0
        for line in f:
            splits = line.strip().split(sep)
            lines.append((int(splits[0]), splits[1]))
            if i >= MAX_LINES - 1:
                break
            i += 1
    return lines


def binary_search(array, element):
    """
    Not used yet
    :param array:
    :param element:
    :return:
    """
    if len(array) == 0:
        return False

    mid = len(array) // 2
    if element == array[mid]:
        return True
    elif element > array[mid] and mid + 1 < len(array):
        return binary_search(array[mid + 1:], element)
    elif element < array[mid]:
        return binary_search(array[:mid], element)

    return False


def correct(word):
    """
    Return most likely correction
    :param word: word string to correct
    :return: word string correction
    """
    return FR_DICT.suggest(word)[0]


def correct_and_lemmatize(input_data, drug_names_set):
    """
    Correction and Lemmatization on input data
    :param drug_names_set: set of drug names
    :param input_data: list(id, sentence)
    :return: list(id, list(word))
    """
    stemmer = FrenchStemmer()
    for i in tqdm(range(len(input_data)), desc="Correct and Lemmatize"):
        line = input_data[i]
        sentence = line[1]
        sentence = sentence.lower()
        splits = word_tokenize(sentence, language='french')
        drug_ids = []
        drug_names = []
        for k in range(len(splits)):
            if not re.match('(\w)+', splits[k]):
                continue
            if "'" in splits[k]:
                word = splits[k].split("'")[1]
            else:
                word = splits[k]
            if unidecode(word) in drug_names_set:
                drug_ids.append(k)
                drug_names.append(splits[k])
                continue
            if not FR_DICT.check(splits[k]):
                suggestions = FR_DICT.suggest(splits[k])
                if suggestions:
                    splits[k] = suggestions[0]
            splits[k] = stemmer.stem(splits[k])
        input_data[i] = (input_data[i][0], sentence,
                         ' '.join(splits), ','.join(map(str, drug_ids)), ','.join(drug_names))
    df = pnd.DataFrame(input_data,
                       columns=(SENTENCE_ID, RAW_SENTENCE_COL, CORR_LEMM_SENTENCE_COL, DRUG_IDS_COL, DRUG_NAMES_COL))
    return df


def main(_args):
    remove_blacklisted_words_from_dict(FR_DICT_BLACKLIST)
    requirements()

    input_data = read_lines(INPU_DATA_FILENAME, sep=';')

    drug_names_path = get_drug_names_path()
    if not os.path.exists(drug_names_path):
        drug_names_df = extract_drug_names(RCP_FILENAME, sep_regex=r'\t|,', id_col_idx=0, name_col_idx=1)
        create_dir(drug_names_path)
        drug_names_df.to_csv(drug_names_path)
    else:
        drug_names_df = pnd.read_csv(drug_names_path)

    drug_names_set = set(drug_names_df[DRUG_NAME_COL])

    corr_lemm_data = correct_and_lemmatize(input_data, drug_names_set)

    count = 0

    for i in range(corr_lemm_data.shape[0]):
        # # To print results
        # print(corr_lemm_data.loc[i, RAW_SENTENCE_COL])
        # print(corr_lemm_data.loc[i, CORR_LEMM_SENTENCE_COL])
        # print(corr_lemm_data.loc[i, DRUG_NAMES_COL])
        # print()

        if corr_lemm_data.loc[i, DRUG_NAMES_COL]:
            count += 1

    corr_lemm_path = get_corr_lemm_path()
    create_dir(corr_lemm_path)
    corr_lemm_data.to_csv(corr_lemm_path)

    # To check drug names, will leave it for a while for testing
    # for i in drug_names_df.index:
    #     if 'trim' in drug_names_df.loc[i, DRUG_COMPLETE_NAME_COL]:
    #         print(drug_names_df.loc[i])
    print(count / MAX_LINES)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
