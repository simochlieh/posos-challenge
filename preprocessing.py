import argparse
from enchant import Dict
import re

FR_DICT = Dict("fr_FR")
DRUG_NAMES_BLACKLIST = ('\n',)

parser = argparse.ArgumentParser(description='This description is shown when -h or --help are passed as arguments.')
parser.add_argument('--required_0',
                    type=int,
                    choices=[1, 2, 3],
                    required=True,
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


def extract_drug_names(filename):
    drug_names = []
    with open(filename, encoding='ISO-8859-1') as f:
        for line in f:
            splits = line.split('\t')
            if len(splits) > 3 and splits[3] not in DRUG_NAMES_BLACKLIST:
                try:
                    drug_names.append(splits[3])
                except:
                    print(line)
                    raise Exception
    return drug_names


def read_lines(filename):
    lines = []
    with open(filename, encoding='utf-8') as f:
        f.readline()
        for line in f:
            splits = line.strip().split(';')
            lines.append((int(splits[0]), splits[1]))
    return lines


def binary_search(array, element):
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


def main(_args):
    print(_args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
