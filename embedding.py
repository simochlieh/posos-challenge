# Drug names embedding file. The goal is to cluster drug names according to their therapeutic indications
import tqdm
import pandas
import requests
import os
from bs4 import BeautifulSoup
import numpy as np

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from functools import partial
import re
from unidecode import unidecode
import sys


FORBIDDEN_INDICATIONS = [" Pas d'indication thérapeutique "]


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

    indication = re.sub(FORBIDDEN_INDICATIONS[0] + "|" + FORBIDDEN_INDICATIONS[1], 'NF', indication)
    indication = re.sub("Plus d'information en cliquant ici", '', indication)
    indication = re.sub(
        ' Vous trouverez les indications thérapeutiques de ce médicament dans le paragraphe 4.1 du RCP ou dans le paragraphe 1 de la notice. Ces documents sont disponibles en cliquant ici ',
        'Ext', indication)

    return indication


###########################################
# Raises approx:
#  16066 "NF" i.e. drug id not in open-medicaments database
#  652 " Pas d'indication therapeutique "
#  1217 " Vous trouverez les indications therapeutiques de ce medicament dans le paragraphe 4.1 du RCP ou dans le paragraphe 1 de la notice. Ces documents sont disponibles en cliquant ici "
# Last one could lead to pdf scraping if really necessary
###########################################

class MyKMeans(KMeans):

    def __init__(self, **kwargs):
        super(MyKMeans, self).__init__(self, **kwargs)

    def __fit(self, X, y=None):
        super(MyKMeans, self).fit(X=X,y=None)



class Tokenizer(TfidfVectorizer, MyKMeans):

    def __init__(self, drug_descriptions='./results/indications', **kwargs):
        self._descriptions = pandas.read_csv(drug_descriptions)

        MyKMeans.__init__(self)
        TfidfVectorizer.__init__(self)

    def fit(self, df, y=None):
        # df is here the full dataframe with sentences, drugs_names etc
        print('fitting')
        #print(df)
        try:
            if set(df.drug_names.unique()) < set(self._descriptions.drug_names):
                raise Exception('Some drugs are not in the description file.')
        except ValueError:
            print('Dimensions mismatch, probably the description file is not the right one.')

        TfidfVectorizer.fit(self, raw_documents=self._descriptions.descriptions, y=y)

    # SpectralClustering.fit(vectorized, **kwargs)

    def transform(self, df, **kwargs):
        vectorized = TfidfVectorizer.transform(self, raw_documents=self._descriptions.descriptions)
        print('vectorized')
        # SpectralClustering.fit(self,X=vectorized, **kwargs)
        # print('sp fit')
        print(self.n_clusters)
        MyKMeans.fit(self, X=vectorized)

        labels = MyKMeans.predict(self, X=vectorized)

        tokens = list(map(lambda l: 'TOKEN_' + str(l), labels))

        mapping = dict(zip(self._descriptions.drugs_names, tokens))

        sentences = df.apply(
            lambda row: re.sub(row['drug_names'], mapping[row['drug_names']], row['corr_lemm_sentence']))

        return sentences


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


def pick_n_clusters(indications, range_n_clusters=[2, 4, 8, 6, 10], algo=SpectralClustering):
    print('Iterating over ' + str(range_n_clusters))
    if type(algo) == str:
        algo = getattr(sklearn.cluster, algo)

    cal_scores = []
    silh_scores = []
    for n_clusters in tqdm.tqdm(range_n_clusters):
        try:
            clusterer = algo(n_clusters,
                             affinity='precomputed')  # algo(n_clusters=n_clusters, affinity=metric, linkage=linkage)
        except:
            clusterer = algo(n_clusters)
        aff = cosine_similarity(indications)
        labels = clusterer.fit_predict(aff)

        cal_scores.append(round(calinski_harabaz_score(indications, labels), 1))
        silh_scores.append(round(silhouette_score(indications, labels, metric='cosine'), 2))

    print('Algorithm: ' + str(algo))
    print('cal_scores: ' + str(cal_scores))
    print('silh_scores: ' + str(silh_scores))

    f, ax = plt.subplots()
    ax.plot(list(range_n_clusters), silh_scores, color='r')
    ax2 = ax.twinx()
    ax2.plot(list(range_n_clusters), cal_scores, color='b')
    f.show()


def filter_indications(serie):
    serie = serie.apply(lambda x: 'NF' if x in FORBIDDEN_INDICATIONS else x)
    serie = serie.apply(lambda s: re.sub("Plus d'information en cliquant ici", '', s))
    return serie


def assess_DBSCAN(X, sup=20):
    print('Assesing DBSCAN with 1/%ilong step' % (sup))
    clusters = []
    s_scores = []
    c_scores = []
    for i in tqdm.tqdm(range(1, sup)):
        algo = DBSCAN(eps=i / sup, metric='cosine')
        labels = algo.fit_predict(X)
        try:
            clusters.append(len(np.unique(algo.labels_)))
            s_scores.append(round(silhouette_score(X, labels, metric='cosine'), 2))
            c_scores.append(calinski_harabaz_score(X, labels))
        except:
            pass

    f, ax = plt.subplots()
    ax.plot(clusters, s_scores, color='r')
    ax.set_ylabel('Silhouette_score', color='r')
    ax2 = ax.twinx()
    ax2.plot(clusters, c_scores, color='b')
    ax1.set_ylabel('C_score', color='b')
    f.show()

    return np.vtack((cluster, s_scores, c_scores)).transpose()


def produce_mapping(s):
    serie = filter_indications(s)
    serie = serie[serie != 'NF']

    algo = SpectralClustering(n_clusters=60, affinity='precomputed')

    vecto = TfidfVectorizer(max_df=0.3, sublinear_tf=True)  # hand tested
    vectorized = vecto.fit_transform(serie).toarray()
    sim = cosine_similarity(vectori)
    labels = algo.fit_predict(sim)
    mapping = {}
    for (m, i) in tqdm.tqdm(enumerate(serie)):
        mapping[i] = 'TOKEN_' + str(labels[m])
    mapping['NF'] = 'TOKEN_NF'

    return mapping


if __name__ == '__main__':
    # drugs = pandas.read_csv('/Users/remydubois/Documents/perso/repos/rd_repos/posos-challenge/results/drug_names')
    # drugs = complete_df(drugs)
    input_train = pandas.read_csv(
        '/Users/remydubois/Documents/perso/repos/rd_repos/posos-challenge/results/corr_lemm/final/input_train')
    toke = Tokenizer()
    toke.fit(df=input_train)
    # print('fit')
    sentences = toke.transform(df=input_train)
    # frame =  complete(train)

    """
    # Take care of deduplicated indications:
    # Replace 'Pas d'inidcation…' by 'NF'
    to_replace = [" Pas d'indication thérapeutique ", ' Vous trouverez les indications thérapeutiques de ce médicament dans le paragraphe 4.1 du RCP ou dans le paragraphe 1 de la notice. Ces documents sont disponibles en cliquant ici ']
    drugs.indicationsTherapeutiques = drugs.indicationsTherapeutiques.replace(to_replace=to_replace, value='NF')
    drugs['count_per_short_name'] = drugs.groupby('drug_names').indicationsTherapeutiques.transform(lambda x:len(np.unique(x)))
    deduplicated = drugs[drugs.count_per_short_name>1]
    duplicated_drugs = deduplicated.drug_names.drop_duplicates()

    #Now, if a drug returns an indication for some time but none for other, we replace 'NF' by the indication:
    drugs = drugs.apply(lambda row: ro, axis=0)

    #train_data = pandas.merge(drugs, train, on=['drug_ids'])

    #mapping = produce_mapping(indications)

    #mapping = produce_mapping(drugs_indic.indicationsTherapeutiques)

    #indications = drugs_indic['indicationsTherapeutiques']

    #vectorizer = TfidfVectorizer()
    #vectorized = vectorizer.fit_transform(indications)

    #n_clusters_KMeans = pick_n_clusters(indications, algo=KMeans)
    #n_clusters_SpectralClustering = pick_n_clusters(indications, algo=SpectralClustering)
    #n_clusters_AgglomerativeClustering = pick_n_clusters(indications, algo=AgglomerativeClustering)
    """
