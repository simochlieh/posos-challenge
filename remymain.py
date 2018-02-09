#Dimensionality reduction file

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from tfidf import *
from params import *
from tempfile import mkdtemp
from shutil import rmtree, copyfile
from sklearn.externals.joblib import Memory
from datetime import datetime
import os

if __name__=='__main__':

	x_train, y_train = read_file()

	vectorizer = MyVectorizer(**params_tfidf)
	pca = PCA(**params_PCA) #From vizu of pca.pkl performed on full data
	clf = DecisionTreeClassifier() #Just for default
	
	cachedir = mkdtemp()
	memory = Memory(cachedir=cachedir, verbose=10)
	pipe = Pipeline(steps=[('tfidf',vectorizer),('pca',pca),('classifier',clf)], memory = memory)

	params_grid = [
	{
		'classifier':[
			DecisionTreeClassifier(), 
			AdaBoostClassifier(n_estimators=150), 
			GradientBoostingClassifier(n_estimators=150, learning_rate=0.08)]
	},
	{
		'classifier':[SGDClassifier()], #faster than SVC
		'classifier__C':[0.8,1.]
	}
	]

	grid = GridSearchCV(pipe, cv=3, n_jobs=3, param_grid=params_grid, verbose=1)

	grid.fit(x_train, y_train)

	#Now write results down.
	timestamp = str(datetime.now()).split('.')[0]
	os.mkdir('./results/%s'%(timestamp))
	with open('./results/'+timestamp+'/info.txt','w+') as f:
		f.write('##############################################')
		f.write('\nBest accuracy: %f'%(grid.best_score_))
		f.write('\nobtained with:\n'+str(grid.best_params_))
		f.write('\n\nAmong a 3 fold CV test on those params:\n'+str(grid.param_grid))
		f.write('\n\nPreprocessing params in params.py file')
	copyfile('params.py','./results/'+timestamp+'/params.txt')