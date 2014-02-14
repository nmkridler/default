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
	catcols += ['id','loss']#,'f521']
	numcols = [c for c in usecols if c not in catcols]
	f = fio.fileio()
	st = time.time()
	uf = pd.read_csv('featsGBM.txt').feature.values[:200]
	numcols = [u for u in uf if u != 'fLast']#list(uf[1:])

	X = f.loadNumericTrain(usecols=numcols)
	fl = fio.filterByLast('../data/train_v2.csv')
	X = np.column_stack((X,fl))

	catcols = ['f26','f264','f254','f105','f144','f670']
	Z, Ztest = f.loadCategorical(catcols,loadTest=True)

	y = f.loadLabels().loss.values
	print "Training data took %f seconds to load" %(time.time() - st)
	"""	
	lr = Lasso()
	#lr = RandomForestRegressor(**ml.INIT_PARAMS['RandomForestClassifier'])
	lr = LDA()
	lr = Ridge()
	#lr = GradientBoostingClassifier(**ml.INIT_PARAMS['GradientBoostingClassifier'])
	#lr = RandomForestClassifier(**ml.INIT_PARAMS['RandomForestClassifier'])
	print "Average AUC: %f"%ml.holdout(X,y,lr,nFolds=5)
	return

	imp_ = lr.feature_importances_
	indices = np.argsort(imp_)[::-1]
	fout = open('featsRFR.txt','w')
	fout.write('feature,score\n')
	allcols = numcols + ['fLast']
	for feat in indices:
		outstr = '%s,%f\n'%(allcols[feat],imp_[feat])
		fout.write(outstr)
	fout.close()
	return
	"""
	st = time.time()
	clf = LDA()
	#rgr = SVR(kernel='poly')
	rgr = Ridge()
	lr = ml.Cascaded(clf,rgr)
	"""
	lr = LDA()
	feats = ml.greedyReductionAUC(X,y,lr,numcols)
	fout = open('feats.txt','w')
	for feat in feats:
		fout.write(feat+'\n')
	fout.close()

	thresholds = np.linspace(0.3,.7,16)
	bestScore, bestScale, bestDec = 1., 1., 1.
	thresholds2 = [0.3]
	for t1 in thresholds:
		for t2 in thresholds: 
			score = ml.stratHoldout(X,y,lr,mean_absolute_error,
				scaleFactor=t1,decisionThreshold=t2,nFolds=NUMFOLDS,verbose=False)
			if score < bestScore:
				print "Score: %f, Scale: %f, Decision: %f"%(score,t1,t2)
				bestScale, bestDec = t1, t2
				bestScore = score
	print "Score: %f, Scale: %f, Decision: %f"%(bestScore,bestScale,bestDec)
	return

	#print "Average Error: %f"%ml.stratKFoldMix(X,Z,y,lr,mean_absolute_error,nFolds=NUMFOLDS)
	#print "Average Error: %f"%ml.stratHoldout(X,y,lr,mean_absolute_error,nFolds=NUMFOLDS)
	print "Average Error: %f"%ml.stratHoldoutMix(X,Z,y,lr,mean_absolute_error,nFolds=NUMFOLDS)
	print "%d-Fold CV took %f seconds"%(NUMFOLDS,time.time() - st)
	return
	"""
	Xtest = f.loadNumericTest(usecols=numcols)
	fl = fio.filterByLast('../data/test_v2.csv')
	Xtest = np.column_stack((Xtest,fl))
	lr.fit(X,y,thresh=0.4)
	print Xtest.shape
	sub_ = pd.read_csv('../data/sampleSubmission.csv')
	print "Training took %f seconds"%(time.time() - st)
	print sub_.shape
	st = time.time()
	yp = lr.predict_proba(Xtest)[:,1]
	print "predict_proba took %f seconds"%(time.time() - st)

	st = time.time()
	yr = lr.rgr.predict(Xtest)
	print "predict took %f seconds"%(time.time() - st)

	sub_.loss = 0.6*yr*(yp > 0.4) #(lr.predict(Xtest)

	lr.fit(Z,y,thresh=0.4,cutoff=40)
	zp = lr.predict_proba(Ztest)[:,1]
	zr = lr.rgr.predict(Ztest)
	print Ztest[:10,:]
	sub_.loss[(zp > 0.7) & (zr > 40)] = zr[(zp > 0.7) & (zr > 40)]
	sub_.loss[sub_.loss < 0] = 0.
	sub_.to_csv('../data/subs/sub02142014.csv',index=False)

if __name__ == "__main__":
	main()
