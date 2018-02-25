# Drug names embedding file. The goal is to cluster drug names according to their therapeutic indications
import tqdm
import pandas
import requests
import os
from bs4 import BeautifulSoup
import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory

from sklearn.cluster import KMeans
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
    description_file = './results/indications'
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

class Tokenizer(TfidfVectorizer, KMeans):

    def __init__(self, verbose=0):
        self._descriptions = pandas.read_csv('./results/indications')

        KMeans.__init__(self)
        TfidfVectorizer.__init__(self)
        self._verbose = verbose

    def fit(self, df, y=None):
        # df is here the full dataframe with sentences, drugs_names etc
        try:
            if set(df.drug_names.unique()) < set(self._descriptions.drug_names):
                raise Exception('Some drugs are not in the description file.')
        except ValueError:
            print('Dimensions mismatch, probably the description file is not the right one.')

        if self._verbose > 0:
            print('Fitting tokenizer…')

        TfidfVectorizer.fit(self, raw_documents=self._descriptions.descriptions, y=y)

    def transform(self, df, **kwargs):

        vectorized = TfidfVectorizer.transform(self, raw_documents=self._descriptions.descriptions)

        KMeans.fit(self, X=vectorized)

        labels = KMeans.predict(self, X=vectorized)

        if self._verbose > 0:
            print('Labelling drugs…')

        tokens = list(map(lambda l: 'TOKEN_' + str(l), labels))

        mapping = dict(zip(self._descriptions.drug_names, tokens))

        if self._verbose > 0:
            print('Drugs mapped…')

        def replace_in_sentence(row, mapping=mapping):
            try:
                positions = row['drug_ids'].split(',')
            except AttributeError:
                positions = []

            positions = list(map(int, positions))

            try:
                drug_names = row['drug_names'].split(',')
            except AttributeError:
                drug_names = []

            drug_tokens = [mapping.get(unidecode(n)) for n in drug_names]

            sentence = row['corr_lemm_sentence'].strip().split(' ')

            tokenized_sentence = ' '.join(
                map(lambda t: t[1] if t[0] not in positions else drug_tokens.pop(0), enumerate(sentence)))
            return tokenized_sentence

        sentences = []
        loop = df.iterrows() if self._verbose == 0 else tqdm.tqdm(df.iterrows())
        for _, row in loop:
            c = replace_in_sentence(row)
            sentences.append(c)

        # Delete description attribute for memory consumption
        # del self._descriptions

        return np.array(sentences)

    def fit_transform(self, df, y=None):
        self.fit(df, y=y)
        return self.transform(df)

    def set_params(self, **kwargs):
        cluster_params = {k: kwargs[k] for k in kwargs if k in KMeans.__init__.__code__.co_varnames}
        tf_params = {k: kwargs[k] for k in kwargs if k in TfidfVectorizer.__init__.__code__.co_varnames}

        KMeans.set_params(cluster_params)
        TfidfVectorizer.set_params(tf_params)
