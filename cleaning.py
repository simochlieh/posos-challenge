import argparse
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from tqdm import tqdm
import pandas as pnd
import re
from unidecode import unidecode

import params
import utils

# notes: # regarder le set de médicaments extraits + enlever les trucs chelou
         # regarder qques phrases au hasard sans médicaments et voir pk?


parser = argparse.ArgumentParser(description="This script runs the cleaning and lemmatization (if enabled) "
                                             "on the raw data. It outputs a file containing the list of drug names "
                                             "(if it doesn't exist) under results/drug_names and a file containing the "
                                             "cleaned and lemmatized data under results/corr_lemm/%label%/input_train")
parser.add_argument('--label',
                    default='final',
                    help="This parameter is used in the output file path of the cleaned and lemmatized data which is "
                         "results/corr_lemm/%label%/input_train. Default value is 'final'.")

parser.add_argument('-l',
                    '--lemmatization',
                    default='true',
                    help='This flag enables lemmatization.')

parser.add_argument('--max-lines',
                    type=int,
                    default=100000,
                    help='This sets the max number of lines to process from the raw input data.')


def remove_blacklisted_words_from_dict(blacklist):
    """
    Some blacklisted french words
    (e.g. carline is a real word but also a drug name)
    :param blacklist: list of blacklisted words
    :return: Nothing
    """
    for word in blacklist:
        params.FR_DICT.remove(word)


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
                if params.FR_DICT.check(new_word):
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
        for line in tqdm(f, desc='pre-processing drug names', total=params.NB_LINES_DRUGS):
            splits = re.split(sep_regex, line)
            if len(splits) > 3 and splits[name_col_idx] not in params.DRUG_NAMES_BLACKLIST:
                names = re.split(r'\s|,|-|/|\\|_', splits[name_col_idx])
                drug_name = None
                for name in names:
                    if len(name) > 3 and re.match('([a-zA-Z])+', name) and not params.FR_DICT.check(name.lower()):
                        drug_name = name
                        break
                if drug_name:
                    suggestion = suggest_accented(drug_name)
                    if suggestion:
                        drug_name = suggestion
                    drug_names.append((int(splits[id_col_idx]), unidecode(drug_name.lower()), ' '.join(names).lower()))
    drug_names = sorted(list(set(drug_names)), key=lambda x: x[1])
    df = pnd.DataFrame(drug_names, columns=[params.DRUG_ID_COL, params.DRUG_NAME_COL, params.DRUG_COMPLETE_NAME_COL])
    df = df.set_index(params.DRUG_ID_COL)
    return df


def read_raw_input(filename, sep, max_lines):
    """
    Read lines of raw input file
    :param max_lines: max number of lines to process
    :param sep: separator character
    :param filename: string
    :return: array of (id, sentence)
    """
    lines = []
    with open(filename, encoding=params.UTF_8) as f:
        f.readline()
        i = 0
        for line in f:
            splits = line.strip().split(sep)
            lines.append((int(splits[0]), splits[1]))
            if i >= max_lines - 1:
                break
            i += 1
    return lines


def correct(word):
    """
    Return most likely correction
    :param word: word string to correct
    :return: word string correction
    """
    return params.FR_DICT.suggest(word)[0]


def correct_and_lemmatize(input_data, drug_names_set, enable_lemm):
    """
    Correction and Lemmatization on input data
    :param enable_lemm: if True, enables lemmatization
    :param drug_names_set: set of drug names
    :param input_data: list(id, sentence)
    :return: list(id, list(word))
    """
    if enable_lemm:
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
                splits[k] = word
            else:
                word = splits[k]
            if unidecode(word) in drug_names_set:
                drug_ids.append(k)
                drug_names.append(splits[k])
                continue
            if not params.FR_DICT.check(splits[k]):
                suggestions = params.FR_DICT.suggest(splits[k])
                if suggestions:
                    splits[k] = suggestions[0]
            if enable_lemm:
                splits[k] = stemmer.stem(splits[k])
        input_data[i] = (input_data[i][0], sentence,
                         ' '.join(splits), ','.join(map(str, drug_ids)), ','.join(drug_names))
    df = pnd.DataFrame(input_data,
                       columns=(params.SENTENCE_ID, params.RAW_SENTENCE_COL,
                                params.CORR_LEMM_SENTENCE_COL, params.DRUG_POS_COL, params.DRUG_NAMES_COL))
    return df


def main(_args):
    print(_args)
    remove_blacklisted_words_from_dict(params.FR_DICT_BLACKLIST)
    requirements()

    input_data = read_raw_input(params.INPUT_TRAIN_FILENAME, sep=';', max_lines=int(_args.max_lines))

    drug_names_path = utils.get_drug_names_path()
    if not os.path.exists(drug_names_path):
        drug_names_df = extract_drug_names(params.RCP_FILENAME, sep_regex=r'\t|,', id_col_idx=0, name_col_idx=1)
        utils.create_dir(drug_names_path)
        drug_names_df.to_csv(drug_names_path)
    else:
        drug_names_df = pnd.read_csv(drug_names_path)

    drug_names_set = set(drug_names_df[params.DRUG_NAME_COL])

    corr_lemm_data = correct_and_lemmatize(input_data, drug_names_set, utils.string_to_bool(_args.lemmatization))

    count = 0

    for i in range(corr_lemm_data.shape[0]):
        # # To print results
        # print(corr_lemm_data.loc[i, RAW_SENTENCE_COL])
        # print(corr_lemm_data.loc[i, CORR_LEMM_SENTENCE_COL])
        # print(corr_lemm_data.loc[i, DRUG_NAMES_COL])
        # print()

        if corr_lemm_data.loc[i, params.DRUG_NAMES_COL]:
            count += 1

    corr_lemm_path = utils.get_corr_lemm_path(_args.label)
    utils.create_dir(corr_lemm_path)
    corr_lemm_data.to_csv(corr_lemm_path)

    # To check drug names, will leave it for a while for testing
    # for i in drug_names_df.index:
    #     if 'trim' in drug_names_df.loc[i, DRUG_COMPLETE_NAME_COL]:
    #         print(drug_names_df.loc[i])
    print("We found drug names in %.2f%% of sentences." % (count / int(_args.max_lines) * 100))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
