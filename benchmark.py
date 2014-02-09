"""

Beating the Benchmark :::::: Kaggle Loan Default Prediction Challenge.
__author__ : Abhishek

"""

import pandas as pd
import numpy as np
import cPickle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import sklearn.linear_model as lm
import time

def testdata(filename):
	X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

	X = np.asarray(X.values, dtype = float)

	col_mean = stats.nanmean(X,axis=0)
	inds = np.where(np.isnan(X))
	X[inds]=np.take(col_mean,inds[1])
	data = np.asarray(X[:,1:-3], dtype = float)

	return data
	
def data(filename):
	X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

	X = np.asarray(X.values, dtype = float)

	col_mean = stats.nanmean(X,axis=0)
	inds = np.where(np.isnan(X))
	X[inds]=np.take(col_mean,inds[1])

	labels = np.asarray(X[:,-1], dtype = float)
	data = np.asarray(X[:,1:-4], dtype = float)
	return data, labels


def createSub(clf, traindata, labels, testdata):
	sub = 1

	labels = np.asarray(map(int,labels))

	niter = 10
	auc_list = []
	mean_auc = 0.0; itr = 0
	if sub == 1:
		xtrain = traindata#[train]
		xtest = testdata#[test]

		ytrain = labels#[train]
		predsorig = np.asarray([0] * testdata.shape[0]) #np.copy(ytest)

		labelsP = []

		for i in range(len(labels)):
			if labels[i] > 0:
				labelsP.append(1)
			else:
				labelsP.append(0)

		labelsP = np.asarray(labelsP)
		ytrainP = labelsP

		lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 2)
		lsvc.fit(xtrain, ytrainP)
		xtrainP = lsvc.transform(xtrain)
		xtestP =  lsvc.transform(xtest)

		clf.fit(xtrainP,ytrainP)
		predsP = clf.predict(xtestP)

		nztrain = np.where(ytrainP > 0)[0]
		nztest = np.where(predsP == 1)[0]

		nztrain0 = np.where(ytrainP == 0)[0]
		nztest0 = np.where(predsP == 0)[0]

		xtrainP = xtrain[nztrain]
		xtestP = xtest[nztest]

		ytrain0 = ytrain[nztrain0]
		ytrain1 = ytrain[nztrain]

		clf.fit(xtrainP,ytrain1)
		preds = clf.predict(xtestP)

		predsorig[nztest] = preds
		predsorig[nztest0] = 0

		np.savetxt('predictions.csv',predsorig ,delimiter = ',', fmt = '%d')

if __name__ == '__main__':
	filename = '../data/train_v2.csv'
	X_test = testdata('../data/test_v2.csv')

	X, labels = data(filename)
	
	clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

	X = preprocessing.scale(X)	
	X_test = preprocessing.scale(X_test)

	st = time.time()
	createSub(clf, X, labels, X_test)
	print "Submission took %f seconds" % (time.time() - st)

