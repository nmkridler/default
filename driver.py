import numpy as np
import pylab as pl
import pandas as pd
import time

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVR
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge, LinearRegression
import fileio as fio
import ml
reload(fio)
reload(ml)

NUMFOLDS = 5
AUCONLY = False
RFIMPORT = False
GRIDSEARCH = False
TRAINCV = False
#DECISION_THRESHOLD = 0.0332681017613 #2.18651689492e-06
DECISION_THRESHOLD = 0.051846234761
SCALE_FACTOR = 0.85

def main():
	""""""
	hdr = [t for t in open('../data/train_v2.csv','r').readline().split('\n')[0].split(',') if t not in ['loss','id']]
	if False:
		hdr = [t for t in open('../data/train_v2.csv','r').readline().split('\n')[0].split(',') if t not in ['loss','id']]
		f = fio.fileio()
		X = f.loadNumericTrain(usecols=hdr)
		y = f.loadLabels().loss.values
		clf = LDA()
		#clf = Ridge()
		feats = ml.MultiGreedyAUC(X,y,clf,hdr)
		print feats
		return

	# open up file containing the columns we wish to use
	f = fio.fileio()
	fz = fio.fileio()
	st = time.time()
	uf = pd.read_csv('featsGBM.txt').feature.values[:145]
	numcols = [u for u in uf if u != 'fLast']#list(uf[1:])
	X = f.loadNumericTrain(usecols=numcols)
	#xcols = ['f2','f271','f274','f527','f528','f204','f777','f278','f210','f724']
	zcols = ['f621', 'f403', 'f535', 'f367', 'f135', 'f333', 'f727', 'f2', 'f271', 'f527', 'f528']
	#zcols = ['f2','f271','f527','f528']
	Z = fz.loadNumericTrain(usecols=zcols)

	y = f.loadLabels().loss.values
	print "Training data took %f seconds to load" %(time.time() - st)
	
	if AUCONLY:
		lr = LDA()
		print "Average AUC: %f"%ml.stratAUC(X,y,lr,nFolds=5)
		return

	if RFIMPORT:
		lr = RandomForestClassifier(**ml.INIT_PARAMS['RandomForestClassifier'])
		lr.fit(X,y)
		imp_ = lr.feature_importances_
		indices = np.argsort(imp_)[::-1]
		fout = open('featsRFR.txt','w')
		fout.write('feature,score\n')
		for feat in indices:
			outstr = '%s,%f\n'%(numcols[feat],imp_[feat])
			fout.write(outstr)
		fout.close()
		return
	
	clf = LDA()
	rgr = Lasso(alpha=0.09)
	rgr = GradientBoostingRegressor(**ml.INIT_PARAMS['GradientBoostingRegressor'])
	if TRAINCV:
		st = time.time()

		print "Average Error: %f"%ml.stratHoldout2Stage(X,Z,y,clf,rgr,nFolds=NUMFOLDS,scaleFactor=SCALE_FACTOR)
		print "%d-Fold CV took %f seconds"%(NUMFOLDS,time.time() - st)
		return

	# Load the test data	
	Xtest = f.loadNumericTest(usecols=numcols)
	Ztest = fz.loadNumericTest(usecols=zcols)

	#clf = LDA()
	#rgr = Ridge()
	# Fit the data
	clf.fit(Z,y > 3)
	yp = clf.predict_proba(Ztest)[:,1]
	zp = clf.predict_proba(Z)[:,1]

	print Z.shape, Ztest.shape
	rgr.fit(X[zp > DECISION_THRESHOLD,:],y[zp > DECISION_THRESHOLD])
	print X.shape, Xtest.shape
	yr = rgr.predict(Xtest)
	yr[yr < 0] = 0
	yr[yr > 100] = 100.
	print "Training took %f seconds"%(time.time() - st)

	sub_ = pd.read_csv('../data/sampleSubmission.csv')
	sub_.loss = SCALE_FACTOR*yr*(yp > DECISION_THRESHOLD)
	sub_.loss[sub_.loss < 0] = 0.

	# Write to file
	sub_.to_csv('../data/subs/sub03022014_4.csv',index=False)

if __name__ == "__main__":
	main()
