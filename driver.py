import numpy as np
import pylab as pl
import pandas as pd
import time

from sklearn.metrics import mean_absolute_error, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import fileio as fio
import ml
reload(fio)
reload(ml)


CPARAMS = {	'filename':'gbm',
			'calctest':True,
			'traindir':'../data/subs/train/',
			'testdir':'../data/subs/test/',
			'numfolds':5}

LPARAMS = { 'numfolds':5,
			'threshold':0.575714285714}


def classifier(filename='',
				calctest=True,
				traindir='',
				testdir='',
				numfolds=5):
	""""""
	outfile = filename+'.csv'

	# open up file containing the columns we wish to use
	st = time.time()
	f = fio.fileio(usePCA=True)
	cols = ['f274','f727', 'f2', 'f271', 'f527', 'f528']
	X = f.loadNumericTrain(usecols=cols)

	yf = f.loadLabels()
	y = yf.loss.values
	print "Training data took %f seconds to load" %(time.time() - st)
	
	# Train the gradient boosting classifier
	clf = GradientBoostingClassifier(**ml.INIT_PARAMS['GradientBoostingClassifier'])

	st = time.time()
	y_ = ml.stratKFold(X,y,clf,nFolds=numfolds)
	fpr, tpr, thresh = roc_curve(y > 0,y_)

	# Print the scores
	print "AUC: %f"%auc(fpr,tpr)
	print "F1 Score: %f"%ml.maxF1(y > 0,y_)
	print "%d-Fold CV took %f seconds"%(numfolds,time.time() - st)

	yf['loss'] = y_
	yf.to_csv(traindir+outfile,index=False)

	if calctest:
		st = time.time()
		# Load the test data	
		Xtest = f.loadNumericTest(usecols=cols)

		# Fit the data
		clf.fit(X,y > 0)

		sub_ = pd.read_csv('../data/sampleSubmission.csv')
		sub_.loss = clf.predict_proba(Xtest)[:,1]

		# Write to file
		sub_.to_csv(testdir+outfile,index=False)
		print "Test submission took %s seconds" %(time.time() - st)

def lgd(numfolds=5,threshold=0.5,featsfile='featsGBM.txt'):
	""""""
	# open up file containing the columns we wish to use
	f = fio.fileio(usePCA=False)
	st = time.time()

	# Found these with magic
	numcols = pd.read_csv(featsfile).feature.values[:150].tolist()

	X = f.loadNumericTrain(usecols=numcols)
	y = f.loadLabels().loss.values
	print "Training data took %f seconds to load" %(time.time() - st)
	
	rgr = GradientBoostingRegressor(**ml.INIT_PARAMS['GradientBoostingRegressor'])

	# Load the test data	
	Xtest = f.loadNumericTest(usecols=numcols)

	# Open up the train/test files
	bTrain = pd.read_csv('../data/subs/train/gbm.csv')
	bTest = pd.read_csv('../data/subs/test/gbm.csv')
	zp = bTrain.loss.values
	yp = bTest.loss.values

	if True:
		y_ = np.zeros(zp.size)
		Z, p = X[zp > threshold,:], y[zp > threshold]
		y_[zp > threshold] = ml.stratKFold(Z, p, rgr, nFolds=2, classify=False)
		print "CV Error: %f"%mean_absolute_error(y, y_)

	# Train on all, transform to log space
	yy = y[zp > threshold]
	yy = np.log10(yy + 1.)

	rgr.fit(X[zp > threshold,:],yy)

	# predict and transform
	yr = rgr.predict(Xtest)
	yr = 10.**(yr) - 1.
	yr[yr < 0] = 0
	yr[yr > 100] = 100.

	print "Training took %f seconds"%(time.time() - st)

	sub_ = pd.read_csv('../data/sampleSubmission.csv')
	sub_.loss = yr*(yp > threshold)
	sub_.loss[sub_.loss < 0] = 0.

	# Write to file
	sub_.to_csv('../data/subs/testSubmission.csv',index=False)

def main():
	# Run the classifier
	classifier(**CPARAMS)

	# Run the regressor
	lgd(**LPARAMS)

if __name__ == "__main__":
	main()
