# Drug names embedding file. The goal is to cluster drug names according to their therapeutic indications
import tqdm
import pandas
import requests
import os
from bs4 import BeautifulSoup
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from unidecode import unidecode
import sys
import warnings
import argparse
from sklearn.pipeline import Pipeline

warnings.simplefilter(action='ignore', category=FutureWarning)


FORBIDDEN_INDICATIONS = [" Pas d'indication thérapeutique "]

parser = argparse.ArgumentParser(description="This script runs loads the description db (if needed) "
                                            "and clusterise drugs based on their description, "
                                            "finally replaces them with tokens based on their cluster")
parser.add_argument('--n_clusters',
                    default=20,
                    help="This parameter is used to choose the number of clusters in which to classify drugs.")


# Requesting open-medicaments API to get therapeutic indications for each drug
def grab_indications(ide):
    a = requests.get('https://open-medicaments.fr/api/v1/medicaments/%s' % (ide))
    json = a.json()
    try:
        html = json['indicationsTherapeutiques']
        soup = BeautifulSoup(html, 'html.parser')
        indication = str(soup.get_text())
    except KeyError:
        indication = 'NF'

    indication = re.sub(FORBIDDEN_INDICATIONS[0], 'NF', indication)
    indication = re.sub("Plus d'information en cliquant ici", '', indication)
    indication = re.sub(
        ' Vous trouverez les indications thérapeutiques de ce médicament dans le paragraphe 4.1 du RCP ou dans le paragraphe 1 de la notice. Ces documents sont disponibles en cliquant ici ',
        'NF', indication)

    return indication

def complete(df):
    print('Completing database…')
    drugs_in_train = df['drug_names'].apply(lambda s: unidecode(str(s)))
    drugs_in_train = [n for s in drugs_in_train for n in s.split(',')]
    drugs_df = pandas.read_csv('/Users/remydubois/Documents/perso/repos/rd_repos/posos-challenge/results/drug_names')

    dico = {}
    for n in np.unique(drugs_in_train):
        dico[n] = list(drugs_df[drugs_df.drug_names == n].drug_ids)

    remaining = dico.keys()  # list(filter(lambda s: s=='NF', dico.keys()))
    
    for n in tqdm.tqdm(remaining):
        indication = 'NF'
        i = 0
        while indication == 'NF' and i < len(dico[n]):
            try:
                indication = grab_indications(dico[n][i])
                # print(indication)
                i += 1
            except KeyboardInterrupt:
                print('Interrupted, saving retrieved information.')
                frame = pandas.DataFrame(list(dico.items()), columns=['drug_names', 'descriptions'])
                frame.to_csv('/Users/remydubois/Documents/perso/repos/rd_repos/posos-challenge/results/indications',
                             index=True)
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

        dico[n] = indication

    frame = pandas.DataFrame(list(dico.items()), columns=['drug_names', 'descriptions'])
    frame.to_csv('/Users/remydubois/Documents/perso/repos/rd_repos/posos-challenge/results/indications', index=True)
    print('DB completed')

    return frame

# As of now: this object does not handle the 'NF', however they are rejected by the tfidf thanks to 
# the parameter max_df. It means that drugs with no descriptions ('NF' for 'Not Found') have zero-coordinates
# in the frequency space.
# It can be fully inegrated in sklearn's pipeline (see main for test), but does not handle properly (at
# least not tested) gridsearch as parameters are not handled properly (**kwargs might raise errors due to
# dirty inheritance).
class Tokenizer(TfidfVectorizer, KMeans):

    def __init__(self, drug_descriptions='./results/indications', n_clusters = 20, **kwargs):
        self._descriptions = pandas.read_csv(drug_descriptions)

        KMeans.__init__(self, n_clusters=n_clusters)
        TfidfVectorizer.__init__(self, max_df = 0.2)

    def fit(self, df, y=None):
        # df is here the full dataframe with sentences, drugs_names etc
        try:
            if set(df.drug_names.unique()) < set(self._descriptions.drug_names):
                raise Exception('Some drugs are not in the description file.')
        except ValueError:
            print('Dimensions mismatch, probably the description file is not the right one.')

        print('Fitting…')
        TfidfVectorizer.fit(self, raw_documents=self._descriptions.descriptions, y=y)

    def transform(self, df, **kwargs):
        
        vectorized = TfidfVectorizer.transform(self, raw_documents=self._descriptions.descriptions)
        
        KMeans.fit(self, X=vectorized)

        labels = KMeans.predict(self, X=vectorized)

        print('Labelled…')

        tokens = list(map(lambda l: 'TOKEN_' + str(l), labels))

        mapping = dict(zip(self._descriptions.drug_names, tokens))

        print('Mapped…')

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
            
            tokenized_sentence = ' '.join(map(lambda t:t[1] if t[0] not in positions else drug_tokens.pop(0), enumerate(sentence)))
            return tokenized_sentence

        sentences = []
        for _,row in tqdm.tqdm(df.iterrows()):
            c = replace_in_sentence(row)
            sentences.append(c)

        #Delete description attribute for memory consumption
        del self._descriptions

        return sentences

    def fit_transform(self,df, y=None):
        self.fit(df)
        return self.transform(df)


def main(_args):
    input_train = pandas.read_csv(
        './results/corr_lemm/final/input_train')

    _ = complete(input_train)

    #toke = Tokenizer(**vars(_args))

    #this is just a test for integration.
    #pca = TfidfVectorizer()

    #pipe = Pipeline([('t',toke),('pca',pca)])

    #pipe.fit_transform(input_train)
    

if __name__ == '__main__':

    args = parser.parse_args()

    main(args)

