# -*- coding: utf8 -*-

# Drug names embedding file. The goal is to cluster drug names according to their therapeutic indications
import tqdm
import pandas
import requests
import os
from bs4 import BeautifulSoup
import numpy as np
from tempfile import mkdtemp
from shutil import rmtree

from nltk.stem.snowball import FrenchStemmer
from sklearn.externals.joblib import Memory
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import re
from unidecode import unidecode
import sys
import warnings
import argparse
from sklearn.pipeline import Pipeline
import io
from PyPDF2 import PdfFileReader
from pathlib2 import Path
import time
from nltk import word_tokenize
import pickle

import params
from utils import get_results_path, get_embedding_dirpath
from fastText import FastText

warnings.simplefilter(action='ignore', category=FutureWarning)

FORBIDDEN_INDICATIONS = [" Pas d'indication thérapeutique "]


def scrape_pdf(url):
    # As the link provided in open-medicaments redirects toward the EU drug databases, the javascript needs to be
    # scrapped as well in order to reach the right storage URL
    r = requests.get(url, allow_redirects=True)
    # Each drug has a number following "#" in the url
    num_med = re.findall('#[0-9]*', url)[0].replace('#', '')
    text = r.text
    # Mind the '080' which is refered to as '80' in the EU database
    num_med = str(int(num_med))
    # Extract the 'id' which is made out of a datetime and other numbers
    try:
        med_id = re.findall('med_nr\[' + num_med + '\]\ \=\ \"[0-9]*\"', text)[0].split('"')[1]
    except:
        print('drug not in EU database')
        return 'NF'
    # Now reconstitute url of the actual storage
    storage_url = 'http://ec.europa.eu/health/documents/community-register/' + med_id[
                                                                               :4] + '/' + med_id + '/anx_' + med_id[
                                                                                                              8:] + '_fr.pdf'
    # request the actual pdf
    r = requests.get(storage_url)
    # Bare
    time.sleep(0.5)
    f = io.BytesIO(r.content)
    try:
        reader = PdfFileReader(f)
    except:
        # Get thrown when the pdf is not compiled the right way
        print('Error while parsing pdf at url: %s' % (url))
        return 'NF'
    # get text which might be spread on page 2 and 3
    contents = reader.getPage(1).extractText() + reader.getPage(2).extractText()
    try:
        indication = re.findall('(?<=Indications thérapeutiques)(.*)(?=4.2)', contents.replace('\n', ''))[0]
    except IndexError:
        # Some pdf are not readable: they do not seem to contain any text (at least, pypdf does not find any text)
        print('Not readable PDF')
        return 'NF'
    return indication


# Requesting open-medicaments API to get therapeutic indications for each drug
def grab_indications(ide):
    a = requests.get('https://open-medicaments.fr/api/v1/medicaments/%s' % (ide))
    json = a.json()
    try:
        html = json['indicationsTherapeutiques']
        # If the json contains an hyperlink, then go to this hyperlink, otherwise parse the content of 'indicationsTherapeutiques'
        if ' Vous trouverez les indications thérapeutiques de ce médicament dans le paragraphe 4.1 du RCP ou dans le paragraphe 1 de la notice.' in html:
            # Get the hyperlink out of the json
            link = re.search('http:\/\/[a-z.\/\-\#0-9_]*', html)[0]
            indication = scrape_pdf(link)
        else:
            # Parse the html
            soup = BeautifulSoup(html, 'html.parser')
            indication = str(soup.get_text())
    except KeyError:
        indication = 'NF'

    # get rid of some trash
    indication = re.sub(FORBIDDEN_INDICATIONS[0], 'NF', indication)
    indication = re.sub("Plus d'information en cliquant ici", '', indication)

    return indication


def complete(df):
    # Check for existing description file:
    description_file = '%s/indications' % get_results_path()
    # If an existing description file is found, then open it in order to complete it, otherwise create a new empty dataframe.
    if Path(description_file).is_file():
        print('Description file found.')
        existing_indications = pandas.read_csv(description_file, index_col=0)
    else:
        print('No description file found… Creating description file.')
        existing_indications = pandas.DataFrame(columns=['drug_names', 'descriptions'])

    # Get the drugs present in the train dataset
    print('Completing database…')
    drugs_in_train = df['drug_names'].apply(lambda s: unidecode(str(s)))
    drugs_in_train = [n for s in drugs_in_train for n in s.split(',')]
    drugs_df = pandas.read_csv('./results/drug_names')

    # get all the ids related to a drug name, in order to query against open-medicaments with all the ids. If a short name is related to severla
    # ids, then all the description should same (or at least very similar), therefore, I just take the first description that pops out.
    dico = {}
    for n in np.unique(drugs_in_train):
        if str(n) != 'nan':
            dico[n] = list(drugs_df[drugs_df.name == n].id)

    # Get the missing descriptions
    remaining = [k for k in dico.keys() if k not in list(existing_indications.drug_names)]

    # Go through all the short names
    for n in tqdm.tqdm(remaining):
        indication = 'NF'
        i = 0
        # For a short name, ho through all its ids
        while indication == 'NF' and i < len(dico[n]):
            try:
                indication = grab_indications(dico[n][i])
                i += 1

            # If interrupted, save what has been retrieved so far.
            except KeyboardInterrupt:
                print('Interrupted, saving retrieved information.')
                existing_indications.to_csv(
                    './results/indications',
                    index=True)
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

        existing_indications = existing_indications.append(
            pandas.DataFrame([[n, indication]], columns=['drug_names', 'descriptions']))

    # frame = pandas.DataFrame(list(dico.items()), columns=['drug_names', 'descriptions'])
    existing_indications.to_csv('./results/indications',
                                index=True)
    print('DB completed')

    return existing_indications


# As of now: this object does not handle the 'NF', however they are rejected by the tfidf thanks to
# the parameter max_df. It means that drugs with no descriptions ('NF' for 'Not Found') have zero-coordinates
# in the frequency space.
# It can be fully integrated in sklearn's pipeline (see main for test), GridSearchCV is functional and works
# properly.

class Tokenizer(Pipeline):

    def __init__(self, do_clustering=True):
        self.stemmer = FrenchStemmer()
        self._descriptions = pandas.read_csv('./results/indications')
        self.do_clustering = do_clustering
        self._verbose = 0

        super(Tokenizer, self).__init__(steps=[('vecto', TfidfVectorizer()), ('cluster', DBSCAN(metric='cosine'))])

    def fit(self, X, y=None):
        super(Tokenizer, self).fit(self._descriptions.descriptions)

    def transform(self, X):

        labels = self.steps[-1][1].labels_

        tokens = list(map(lambda l: 'TOKEN_' + str(l), labels))

        mapping = dict(zip(self._descriptions.drug_names, tokens))

        def replace_in_sentence(row, stemmer, mapping=mapping):
            try:
                positions = row['drug_ids'].split(',')
            except AttributeError:
                positions = []

            positions = list(map(int, positions))

            try:
                drug_names = row['drug_names'].split(',')
            except AttributeError:
                drug_names = []

            if self.do_clustering:
                replacement = [mapping.get(unidecode(n)) for n in drug_names]
            else:
                replacement = [' '.join([stemmer.stem(word) for word in word_tokenize(mapping.get(unidecode(n)))
                                         if re.match('(\w)+', word)])
                               for n in drug_names]

            sentence = row[params.CORR_LEMM_SENTENCE_COL].strip().split(' ')

            if self.do_clustering:
                tokenized_sentence = ' '.join(
                    map(lambda t: t[1] if t[0] not in positions else replacement.pop(0), enumerate(sentence)))
            else:
                tokenized_sentence = ' '.join(sentence + replacement).lower()
                # print(replacement)
                # print(tokenized_sentence)
            return tokenized_sentence

        sentences = []
        loop = X.iterrows() if self._verbose == 0 else tqdm.tqdm(X.iterrows())
        for _, row in loop:
            c = replace_in_sentence(row, stemmer=self.stemmer)
            sentences.append(c)

        return np.array(sentences)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


def embedde_drugs(indications):
    # drugs is here a list of drug names
    if not type(indications) == pandas.DataFrame:
        raise ValueError('Input should be a df of drug names and descriptions.')
    # Check if an embedding file was  started

    if Path(get_embedding_dirpath()).is_file():
        print('Description file found.')
        with open(get_embedding_dirpath(), 'rb') as f:
            existing_embeddings = pickle.load(f)

    else:
        print('No description file found… Creating description file.')
        existing_embeddings = {}

    # Read in the drug names from indications as it has been completed by the test file. Replace NFs by médicament.
    indications.descriptions = indications['descriptions'].apply(lambda s: s.replace('NF', 'médicament'))

    # Now perform tfidf in order to weight the embedding for each word in each description
    # vecto = TfidfVectorizer(max_df=0.3)
    # vectorized = vecto.fit_transform(indications.descriptions)

    # model = FastText.load_model('./wiki.fr/wiki.fr.bin')
    remainings = list(set(indications.drug_names) - set(existing_embeddings.keys()))
    try:
        for d in tqdm.tqdm(indications[indications.drug_names.isin(remainings)].itertuples(), desc='embedding…'):
            existing_embeddings.update({d.drug_names: model.get_sentence_vector(d.descriptions.lower())})
    except NameError:
        print('Model not loaded yet, uncomment line 259 to do so (5gig RAM required).')
    with open(get_embedding_dirpath(), 'wb') as f:
        pickle.dump(existing_embeddings, f)


if __name__ == '__main__':
    indications = pandas.read_csv('./results/indications', index_col=0)
    embedde_drugs(indications)
