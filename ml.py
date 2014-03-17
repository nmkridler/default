import numpy as np
import pandas as pd
import pylab as pl

from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from multiprocessing import Pool

INIT_PARAMS = {

	"GradientBoostingClassifier":{'max_depth':7, 'subsample':0.5,
		'verbose':0, 'random_state':1337,
		'min_samples_split':5, 'min_samples_leaf':5, 'max_features':4,
		'n_estimators': 125, 'learning_rate': 0.1
		},
	"GradientBoostingRegressor":{'loss':'lad','subsample':0.5,
          'max_depth':5,'min_samples_split':85,
          'min_samples_leaf':85,'max_features':50,'learning_rate':0.1,
          'n_estimators':500,'alpha':0.50,'random_state':1337}
}


def stratKFold(X,y,clf,nFolds=5,seed=1337,frac=1.,classify=True):
	"""
		Perform stratified k-fold cross validation

		Args:
			X: features
			y: labels
			clf: sklearn classifier object
			nFolds: number of folds
			seed: random seed
			frac: fraction of h0 samples train with
				this is to undersample

		Returns:
			probabilities

	"""
	np.random.seed(seed)
	kf = StratifiedKFold(y > 0, n_folds=nFolds)
	yp = np.empty(y.size)

	for train, test in kf:
		i0, i1 = train[y[train] == 0], train[y[train] > 0]

		# Undersample if necessary
		numTrain = int(frac*len(i0))
		if numTrain > 0:
			np.random.shuffle(i0)
			ind_ = np.concatenate((i0[:numTrain],i1))
		else:
			ind_ = i1.copy()		

		# Classify or Regress?
		if classify:
			clf.fit(X[ind_,:],y[ind_] > 0)
			yp[test] = clf.predict_proba(X[test,:])[:,1]
		else:
			# Transform to improve distribution
			yy = np.log10(1. + y[ind_])
			clf.fit(X[ind_,:], yy)
			yp[test] = (10.**clf.predict(X[test,:]) - 1.)
	
	return yp

def holdout(X,y,rgr,fraction=0.2,nFolds=10,seed=1337,verbose=True):
	""""""
	meanScore = 0.
	for i in xrange(nFolds):
		xTr, xCV, yTr, yCV = train_test_split(X,y,
			test_size=fraction,random_state=i*seed)
		rgr.fit(xTr,yTr)
		y_ = rgr.predict(xCV)
		y_[y_ < 0] = 0.
		thisScore = mean_absolute_error(yCV[yCV >0],y_[yCV > 0])
		meanScore += thisScore
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)

	meanScore /= nFolds
	return meanScore

def maxF1(t,p):
	"""
		Determine the maximum F1 score
		Args:
			t: truth
			p: predictions

		Returns:
			max f1 score
	"""
	precision, recall, thresholds = precision_recall_curve(t,p)
	denom = (precision + recall)
	f1 = 2.*precision*recall/(precision + recall)
	f1[denom == 0] = 0.
	return f1.max()

def maxMAE(t,p,r,minval=0,maxval=1,nThresh=4,maxIter=6):
	""" 
		Do a weird bracketed search for the best MAE
		note: this is an analysis tool

		Args:
			t: truth
			p: prediction:
			r: regression model
			minval: minimum threshold for search
			maxval: maximum threshold for search
			nThresh: number of values to search
			maxIter: depth of search

		Returns:
			maximum score

	"""
	thresh_ = np.linspace(minval,maxval,nThresh)
	maxScore, iter_ = 1., 0
	while iter_ < maxIter:
		fs = [mean_absolute_error(t,r*(p > th)) for th in thresh_]
		amax, tmax = np.argmin(fs), np.min(fs)
		if tmax < maxScore:
			maxScore = tmax
		if tmax > maxScore:
			break
		minval, maxval = np.max([0,amax-1]),np.min([nThresh-1,amax+1])
		nThresh *= 2
		thresh_ = np.linspace(thresh_[minval],thresh_[maxval],nThresh)
		iter_ += 1

	print thresh_[amax]
	return maxScore	

def stratKFoldF1(X,y,clf,frac=1.,nFolds=5,seed=1337):
	"""
		Run stratified K-fold

		Args:
			X: features
			y: labels
			clf: sklearn classifier object
			nFolds: number of folds
			seed: random seed
			frac: fraction of h0 samples train with
				this is to undersample

		Returns:
			f1 score		

	"""
	yp = stratKFold(X,y,clf,nFolds=nFolds,seed=seed,frac=frac,classify=True)
	return maxF1(y, yp)

def stratKFoldAUC(X,y,clf,frac=1.,nFolds=5,seed=1337,verbose=True):
	"""
		Run stratified K-fold

		Args:
			X: features
			y: labels
			clf: sklearn classifier object
			nFolds: number of folds
			seed: random seed
			frac: fraction of h0 samples train with
				this is to undersample

		Returns:
			auc score		
	"""
	yp = stratKFold(X,y,clf,nFolds=nFolds,seed=seed,frac=frac,classify=True)
	fpr, tpr, thresh = roc_curve(y>0,yp)
	return auc(fpr,tpr)

def stratAUC(X,y,clf,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	"""
		Repeated stratified holdouts

		Args:
			X: features
			y: labels
			clf: sklearn classifier object
			nFolds: number of folds
			fraction: size of test set
			seed: random seed
			verbose: flag for printing

		Returns:
			auc score		

	"""
	meanScore = 0.
	yTarget = 1*(y > 0)
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(X[train,:],yTarget[train])
		y_ = clf.predict_proba(X[test,:])[:,1]
		fpr, tpr, thresh = roc_curve(yTarget[test],y_)
		thisScore = auc(fpr,tpr)
		if verbose:
			print "AUC: %f"%thisScore
		meanScore += thisScore

	return meanScore/nFolds

def featureScoreF1(x):
	""""""
	X, y, clf = x
	pca = PCA(n_components=X.shape[1])
	X = pca.fit_transform(X)
	return stratKFoldF1(X,y,clf,verbose=False)

def featuresScoreAUC(x):
	""""""
	X, y, clf = x
	pca = PCA(n_components=X.shape[1])
	X = pca.fit_transform(X)
	return stratKFoldAUC(X,y,clf,verbose=False)

def MultiGreedyF1(X,y,clf,fnames):
	"""
		Do a greedy search maximizing F1

		Args:
			X: features
			y: labels
			clf: sklearn classifier
			fnames: column names
	"""	
	pool = Pool(processes=4)
	cutoff = 0
	bestFeats = [271,1,717,268,520,521]
	lastScore = 0
	allFeats = [f for f in xrange(X.shape[1])]

	while True:
		testFeatSets = [[f] + bestFeats for f in allFeats if f not in bestFeats]
		args = [(X[:,fSet],y,clf) for fSet in testFeatSets]
		scores = pool.map(featureScoreF1,args)
		(score, featureSet) = max(zip(scores,testFeatSets))
		print featureSet
		print "Max AUC: %f"%score
		if score < lastScore:
			break
		lastScore = score
		bestFeats = featureSet

	pool.close()
	return [fnames[i] for i in bestFeats]

def MultiGreedyAUC(X,y,clf,fnames):
	"""
		Do a greedy search maximizing AUC

		Args:
			X: features
			y: labels
			clf: sklearn classifier
			fnames: column names
	"""
	pool = Pool(processes=4)

	bestFeats = []
	lastScore = 0
	allFeats = [f for f in xrange(X.shape[1])]

	while True:
		testFeatSets = [[f] + bestFeats for f in allFeats if f not in bestFeats]
		args = [(X[:,fSet],y,clf) for fSet in testFeatSets]
		scores = pool.map(featureScoreAUC,args)
		(score, featureSet) = max(zip(scores,testFeatSets))
		print featureSet
		print "Max AUC: %f"%score
		if score < lastScore:
			break
		lastScore = score
		bestFeats = featureSet

	pool.close()
	return [fnames[i] for i in bestFeats]