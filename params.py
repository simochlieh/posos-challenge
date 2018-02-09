# Parameters file for experiments
# Allows easier modifications and access to the parameters:
# Once the main is written, only this file needs to be modified in order to
# Proceed to various experiments

#################################################
# TF-IDF parameters
#################################################

params_tfidf =	{
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
	'sublinear_tf':False,
	'forbidden_words':['autisme'],
	'magic_word':''}

#################################################
# other parameters ...
#################################################
