## This is the tf-idf script for the preprocessing step.
# it should be tested in the queue of the cleansing step.
# The read_file func is pointless as it wont be used in real framework.
# The sklearn tfidfvectorizer class was subclassed in order to integrate
# The whole process into a sklearn pipeline (very convenient for CV grid testing)
# This object stores the parameters it was fed with, even if those parameters are accessible
# Through the params.py file.

#Throws a weird deprecation error btw

#TF-file for preprocessing step
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import re
from params import params_tfidf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
		questions = map(lambda s:re.sub('[-?\']',' ',s), questions)
		#Strip spaces
		questions = map(lambda s:s.strip(), questions)

	return questions


#Needed to build a class for better integration
class MyVectorizer(TfidfVectorizer):
	
	# In practice, 
	# all the sklearn parameters will be fed into this __init__ from the params.py file.
	def __init__(self, forbidden_words=[], magic_word='',**kwargs):
		self.forbidden_words = forbidden_words
		self.magic_word = magic_word

		#Get params
		params = {'forbidden_words':forbidden_words,'magic_word':magic_word}
		params.update(kwargs)
		self._params = params

		#Init mother
		super(MyVectorizer, self).__init__(self, params)

	def fit(self, sentences):
		#Filter words
		if not self.forbidden_words==[]:
			sentences = map(lambda s:' '.join([w if w not in self.forbidden_words else self.magic_word for w in s.split()]), sentences)
		print('Fitting…')
		super(MyVectorizer, self).fit(sentences)

	# Don't know why it needs to be overriden. It looks like if not overriden, fit_transform calls
	# the mother's method instead of the child one, which is weird.
	def transform(self,sentences):
		print('Transforming…')
		return super(MyVectorizer, self).transform(sentences)

	# Same remark	
	def fit_transform(self,sentences):
		sentences = list(sentences)
		self.fit(sentences)
		out = self.transform(sentences)
		print('Vectorized %i sentences for a total of %i words'%out.shape)
		return out

	# Protect
	@property
	def params(self):
		return self._params



if __name__=='__main__':
	args = parser.parse_args()
	
	questions = read_file()
	params_tfidf.update(vars(args))
	m = MyVectorizer(**params_tfidf)
	vectorized = m.fit_transform(questions)
	print(m.params)