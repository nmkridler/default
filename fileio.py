import pandas as pd
import numpy as np

import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class fileio(object):
	"""
	"""
	def __init__(self,
				 usePCA = True,
				 trainfile='../data/train_v2.csv',
				 testfile='../data/test_v2.csv'):

		self.trainfile = trainfile
		self.testfile = testfile
		self.enc = OneHotEncoder()
		self.scale = StandardScaler()
		self.numAvg = {}
		self.usePCA = usePCA

	def loadLabels(self):
		"""
		"""
		return pd.read_csv(self.trainfile,usecols=['id','loss'])

	def loadNumericTrain(self,usecols=None):
		"""
		"""
		df = pd.read_csv(self.trainfile,usecols=usecols+['id'])
		self.pca = PCA(n_components=len(usecols))

		#impute and scale
		for c in usecols:
			df[c] = df[c].astype('float') # convert to float
			self.numAvg[c] = df[c].mean()    # get the average
			df[c] = df[c].fillna(df[c].mean()) # impute
		
		X = np.array(df.ix[:,usecols])
		X = self.scale.fit_transform(X)
		if self.usePCA:
			X = self.pca.fit_transform(X)
		return X

	def loadNumericTest(self,usecols=None):
		"""
		"""
		if len(self.numAvg.keys()) == 0:
			print "Train data hasn't been loaded"
			return

		df = pd.read_csv(self.testfile,usecols=usecols+['id'])

		#impute and scale
		for c in usecols:
			df[c] = df[c].astype('float')
			df[c] = df[c].fillna(self.numAvg[c])

		X = np.array(df.ix[:,usecols])
		X = self.scale.transform(X)	
		if self.usePCA:
			X = self.pca.transform(X)
		return X
