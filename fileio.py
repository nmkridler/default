import pandas as pd
import numpy as np

import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

def relabel(df,key,threshold=5):
	""" One-Hot encode the categorical variables
	"""
	y = df[key][df.set == 0].value_counts() # Take only the training data
	index = np.arange(y.size,dtype='int32') # Create an array of integers
	valid = np.sum(y > threshold)           # threshold
	index[valid:] = valid                   

	# Merge with base data frame
	df = pd.merge(df,
				  pd.DataFrame({key:y.index,key+'cat':index}),
				  how='inner',on=key,sort=False)

	# Impute missing values
	df[key+'cat'] = df[key+'cat'].fillna(df[key][df.set == 0].mean())
	return df

class fileio(object):
	"""
	"""
	def __init__(self,
				 trainfile='../data/train_v2.csv',
				 testfile='../data/test_v2.csv'):

		self.trainfile = trainfile
		self.testfile = testfile
		self.enc = OneHotEncoder()
		self.scale = MinMaxScaler()
		self.numAvg = {}

	def loadLabels(self):
		"""
		"""
		return pd.read_csv(self.trainfile,usecols=['id','loss'])

	def loadNumericTrain(self,usecols=None):
		"""
		"""
		df = pd.read_csv(self.trainfile,usecols=usecols+['id'])

		#impute and scale
		for c in usecols:
			df[c] = df[c].astype('float') # convert to float
			self.numAvg[c] = df[c].mean()    # get the average
			df[c] = df[c].fillna(df[c].mean()) # impute
		
		X = np.array(df.ix[:,usecols])
		X = self.scale.fit_transform(X)
		
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
		return X

	def loadCategorical(self,catcols,loadTest=False):
		"""
		"""
		df = pd.read_csv(self.trainfile,usecols=catcols+['id'])
		df['set'] = 0

		if loadTest:
			tf = pd.read_csv(self.testfile,usecols=catcols+['id'])
			tf['set'] = 1
			df = df.append(tf,ignore_index=False)

		relabeled = []
		for c in catcols:
			df[c] = df[c].fillna(cf[c].max()+1)
			df = relabel(df,c)
			relabeled.append(c+'cat')
			
		df = df.sort_index(by='id')
		train = df.loc[df.set == 0]
		train = self.enc.fit_transform(train.ix[:,relabeled])
		if loadTest:
			test = df.loc[df.set == 1]
			test = self.enc.transform(test.ix[:,relabeled])
		else:
			test = None

		return train, test	



