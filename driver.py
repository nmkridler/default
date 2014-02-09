import numpy as np
import pylab as pl
import pandas as pd
import time

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, ElasticNet
import fileio as fio
import ml
reload(fio)
reload(ml)

NUMFOLDS = 5

def main():
	""""""
	# open up file containing the columns we wish to use
	usecols = [line.split('\r\n')[0] for line in open('usableCols.txt').readlines()]
	catcols = ['f776','f777','f679','f725']
	catcols += ['f83','f93','f103','f113','f123']
	catcols += ['f152','f162','f172','f182','f191']
	catcols += ['f222','f232','f242','f252','f262']
	catcols += ['f291','f299','f307','f315','f323']
	catcols += ['f2','f4','f5','f6','f73','f403','f77']
	catcols += ['id','loss']
	numcols = [c for c in usecols if c not in catcols]
	f = fio.fileio()
	st = time.time()
	X = f.loadNumericTrain(usecols=numcols)
	y = f.loadLabels().loss.values
	print "Training data took %f seconds to load" %(time.time() - st)
	print X.shape

	st = time.time()
	clf = LDA()
	rgr = Ridge()
	lr = ml.Cascaded(clf,rgr)
	print "Average Error: %f"%ml.stratHoldout(X,y,lr,mean_absolute_error,nFolds=NUMFOLDS)
	print "%d-Fold CV took %f seconds"%(NUMFOLDS,time.time() - st)
	return
	Xtest = f.loadNumericTest(usecols=numcols)
	lr.fit(X,y)
	print Xtest.shape
	sub_ = pd.read_csv('../data/sampleSubmission.csv')
	print sub_.shape
	sub_.loss = lr.predict(Xtest)
	sub_.loss[sub_.loss < 0] = 0.
	sub_.to_csv('../data/subs/sub02092014.csv',index=False)

if __name__ == "__main__":
	main()
