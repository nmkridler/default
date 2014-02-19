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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
import fileio as fio
import ml
reload(fio)
reload(ml)

NUMFOLDS = 5
AUCONLY = False
RFIMPORT = False
GRIDSEARCH = False
TRAINCV = False
DECISION_THRESHOLD = 0.2
SCALE_FACTOR = 0.7

def main():
	""""""
	# open up file containing the columns we wish to use
	f = fio.fileio()
	st = time.time()
	uf = pd.read_csv('featsGBM.txt').feature.values[:145]
	numcols = [u for u in uf if u != 'fLast']#list(uf[1:])
	X = f.loadNumericTrain(usecols=numcols)
	#numcols = ['f527','f528']
	#Z = f.loadNumericTrain(usecols=numcols)
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
	rgr = Ridge()
	lr = ml.Cascaded(clf,rgr)

	if GRIDSEARCH:
		thresholds = np.linspace(0.1,.15,16)
		thresholds2 = np.linspace(0.2,1,4)
		bestScore, bestScale, bestDec = 1., 1., 1.
		for t1 in thresholds2:
			for t2 in thresholds: 
				score = ml.stratHoldoutMix(X,Z,y,lr,mean_absolute_error,
					scaleFactor=t1,decisionThreshold=t2,nFolds=NUMFOLDS,verbose=False)
				print "This Score: %f, Scale: %f, Decision: %f"%(score,t1,t2)
				if score < bestScore:
					print "Best Score: %f, Scale: %f, Decision: %f"%(score,t1,t2)
					bestScale, bestDec = t1, t2
					bestScore = score
		print "Score: %f, Scale: %f, Decision: %f"%(bestScore,bestScale,bestDec)
		return

	if TRAINCV:
		st = time.time()
	
		print "Average Error: %f"%ml.stratHoldout(X,y,lr,mean_absolute_error,nFolds=NUMFOLDS,
			scaleFactor=SCALE_FACTOR,decisionThreshold=DECISION_THRESHOLD)
		
		#print "Average Error: %f"%ml.stratKFold(X,y,lr,mean_absolute_error,nFolds=NUMFOLDS)
		print "%d-Fold CV took %f seconds"%(NUMFOLDS,time.time() - st)
		return

	# Load the test data	
	Xtest = f.loadNumericTest(usecols=numcols)

	# Fit the data
	lr.fit(X,y,thresh=0.15)
	print "Training took %f seconds"%(time.time() - st)

	sub_ = pd.read_csv('../data/sampleSubmission.csv')
	st = time.time()
	yp = lr.predict_proba(Xtest)[:,1]
	print "predict_proba took %f seconds"%(time.time() - st)

	st = time.time()
	yr = lr.rgr.predict(Xtest)
	print "predict took %f seconds"%(time.time() - st)

	sub_.loss = SCALE_FACTOR*yr*(yp > DECISION_THRESHOLD)
	sub_.loss[sub_.loss < 0] = 0.

	# Write to file
	sub_.to_csv('../data/subs/sub02182014.csv',index=False)

if __name__ == "__main__":
	main()
