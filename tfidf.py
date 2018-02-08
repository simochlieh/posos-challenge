## This is the tf-idf script for the preprocessing step.
# it should be tested in the queue of the cleansing step.
# The read_file func is pointless as it wont be used in real framework.
# The vectorize func can take as argument any argument of the sklearn tfidfVectorizer __init__ method
# Argparse is not handled the right way for now.

#Throws a weird deprecation error btw

#TF-file for preprocessing step
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import re

parser = argparse.ArgumentParser(description='This description is shown when -h or --help are passed as arguments.')

parser.add_argument('--min_freq',
                    type=float,
                    required=False,
                    default=1.,
                    help='This is the minimum #appearances for a word to reach in order to be maintained')


# Just reads the csv file by getting rid of the header.
# For speed and memory matter, it yields a generator of sentences.
# This is just intended at my own development testing for the below vectorizer
def read_file(path='/Users/remydubois/Desktop/posos/input_test.csv'):
	#Open and get rid of the header
	with open(path,'r') as f:
		questions = f.read().splitlines()[1:]
		#Split and leave the question id for now
		questions = map(lambda s:s.split(';')[1], questions)
		#Filter punctuation
		questions = map(lambda s:re.sub('[-?]',' ',s), questions)
		#Strip spaces
		questions = map(lambda s:s.strip(), questions)

	return questions


# This func applies sklearn tfidfvectorizer fit_transform to the fed sentences.
# Forbidden words can be either a list or a single word
def vectorize(
	sentences, 
	forbidden_words=[],
	magic_word='',
	 **kwargs):

	#default params for tfidf
	params =	{
		'input':"content", 
		'encoding':"utf-8", 
		'decode_error':"strict", 
		'strip_accents':None, 
		'lowercase':True, 
		'preprocessor':None, 
		'tokenizer':None, 
		'analyzer':"word", 
		'stop_words':None, 
		#'token_pattern':"(?u)\b\w\w+\b", 
		'ngram_range':(1, 1), 
		'max_df':1.0, 
		'min_df':1, 
		'max_features':None, 
		'vocabulary':None, 
		'binary':False, 
		#dtype:<class "numpy.int64">, 
		'norm':"l2", 
		'use_idf':True, 
		'smooth_idf':True, 
		'sublinear_tf':False}

	#Use kwargs, feed em to the params dictionary
	params.update(kwargs)
	
	#Create transformer object with the given parameters
	transformer = TfidfVectorizer(**params)

	#Filter the drug names if needed. Replace em with whatever desired: blank or any token.
	#Convert if lazy mistake
	forbidden_words = forbidden_words if hasattr(forbidden_words, '__iter__') else [forbidden_words]
	#Filter
	if not forbidden_words==[]:
		sentences_filtered = map(lambda s:' '.join([w if w not in forbidden_words else magic_word for w in s.split()]), sentences)

	#Vectorize. this return a sparse matrix of dim (#Sentences, Vocabulary size).
	vectorized = transformer.fit_transform(sentences)
	print("Vectorized…\nVocabulary size: %i, and %i sentences"%vectorized.shape)

	#Returns the vectorized sparse matrix AND the params used for vectorization, for better experiment tracking.
	return vectorized, params

if __name__=='__main__':
	args = parser.parse_args()
	
	questions = read_file()

	#Need to parse args
	vectorized, params_tfidf = vectorize(
		sentences=questions)