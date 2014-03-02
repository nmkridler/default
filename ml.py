import numpy as np
import pandas as pd
import pylab as pl

from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, f1_score
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

from multiprocessing import Pool

INIT_PARAMS = {
	"LogisticRegression": { 
		'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':.010, 
		'fit_intercept':True, 'intercept_scaling':1.0, 
        'class_weight':None, 'random_state':1337
        },
    "RandomForestClassifier": {
    	'max_depth':128,'min_samples_split':100,'min_samples_leaf':100,
		'n_jobs':1,'verbose':True,'n_estimators':50,'random_state':1337
		},
	"SGDClassifier":{
		'loss':'log','penalty':'elasticnet','alpha':1.0,'n_iter':20,
		'shuffle':True,'random_state':1337,'class_weight':None
		},
	"GradientBoostingClassifier":{'max_depth':8, 'subsample':0.5,
		'verbose':2, 'random_state':1337,
		'min_samples_split':100, 'min_samples_leaf':100, 'max_features':10,
		'n_estimators': 125, 'learning_rate': 0.1},
	"GradientBoostingRegressor":{'loss':'lad','subsample':0.5,
          'max_depth':6,'min_samples_split':85,
          'min_samples_leaf':85,'max_features':50,'learning_rate':0.05,
          'n_estimators':500,'alpha':0.50,'random_state':1337}
}


class Cascaded(object):
	"""
	"""
	def __init__(self,classifier,regressor):
		self.clf = classifier
		self.rgr = regressor

	def fit(self,X,y,thresh=0.5,cutoff=0):
		""""""
		self.clf.fit(X,(y > cutoff))
		#self.rgr.fit(X[y > 0],y[y > 0])
		#yp = self.clf.predict(X)
		#self.rgr.fit(X[yp > 0],y[yp > 0])
		yp = self.clf.predict_proba(X)[:,1]
		self.rgr.fit(X[yp > thresh],y[yp > thresh])

	def predict(self,X):
		""""""
		return self.rgr.predict(X)
	
	def predict_proba(self,X):
		""""""
		return self.clf.predict_proba(X)

def stratHoldout(X,y,clf,scoreFunc,scaleFactor=0.6,decisionThreshold=0.2,
	nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore, i = 0., 1
	allZero = 0.
	yTarget = 1*(y > 0)
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(X[train,:],y[train],thresh=0.15)
		yp = clf.predict_proba(X[test,:])[:,1]
		yr = clf.predict(X[test,:])

		#y_ = yr*(yp > decisionThreshold)*scaleFactor
		y_ = (yp > decisionThreshold)*scaleFactor
		y_[y_ < 0] = 0.
		thisScore = scoreFunc(y[test],y_)
		thisZero = scoreFunc(y[test],np.zeros(len(test)))
		if verbose:
			yp = clf.predict_proba(X[test,:])[:,1]
			fpr, tpr, thresh = roc_curve(yTarget[test],yp)
			print "AUC: %f"%auc(fpr,tpr)
			print "Target Samples: %d"%np.sum(yTarget[test])
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(y[test],np.zeros(len(test))))
			#pl.plot(fpr,tpr,lw=2)
			#pl.show()
			#cm = confusion_matrix(yTarget[test],y_)
			#print "Confusion Matrix: "
			#print cm
		meanScore += thisScore
		allZero += thisZero
		i += 1

	if verbose:
		print "All Zero Average Score: %f"%(allZero/nFolds)
	return meanScore/nFolds

def stratHoldout2Stage(X,Z,y,clf,rgr,scaleFactor=0.68,decisionThreshold=0.051846234761,
	nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore, i = 0., 1
	allZero = 0.
	yTarget = 1*(y > 0)
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(Z[train,:],y[train] > 3)
		zp = clf.predict_proba(Z[train,:])[:,1]
		yp = clf.predict_proba(Z[test,:])[:,1]

		rgr.fit(X[train[zp > decisionThreshold],:],y[train[zp > decisionThreshold]])
		#rgr.fit(X[train[y[train] > 0],:],y[train[y[train] > 0]])
		yr = rgr.predict(X[test,:])
		yr[yr > 100] = 100.
		yr[yr < 0] = 0.

		y_ = yr*(yp > decisionThreshold)*scaleFactor
		y_[y_ < 0] = 0.
		thisScore = mean_absolute_error(y[test],y_)
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
		meanScore += thisScore
		i += 1

	return meanScore/nFolds


def stratKFoldDF(X,y,clf,nFolds=5,seed=1337,verbose=True,cutoff=0):
	""""""
	kf = StratifiedKFold(y > cutoff,n_folds=nFolds)
	yp, y_ = np.empty(y.size), np.empty(y.size)
	meanScore, i = 0., 1
	for train, test in kf:
		clf.fit(X[train,:],y[train] > cutoff)
		#yp[test] = clf.decision_function(X[test,:])
		yp[test] = clf.predict_proba(X[test,:])[:,1]
	
	return yp

def stratKFoldR(X,y,clf,frac=1.,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y > 0,n_folds=nFolds)
	yp, y_ = np.empty(y.size), np.empty(y.size)
	np.random.seed(seed)
	for train, test in kf:
		i0, i1 = train[y[train] == 0], train[y[train] > 0]
		numTrain = int(frac*len(i0))
		if numTrain > 0:
			np.random.shuffle(i0)
			ind_ = np.concatenate((i0[:numTrain],i1))
		else:
			ind_ = i1.copy()
		clf.fit(X[ind_,:],y[ind_])
		yp[test] = clf.predict(X[test,:])
	
	return yp

def stratKFold(X,y,clf,scoreFunc,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y > 0,n_folds=nFolds)
	yp, yr, y_ = np.empty(y.size), np.empty(y.size), np.empty(y.size)
	meanScore, i = 0., 1
	for train, test in kf:
		clf.clf.fit(X[train,:],y[train] > 0)
		yp[test] = clf.predict_proba(X[test,:])[:,1]
		#yr[test] = clf.predict(X[test,:])
		#y_[test] = 0.6*yr[test]*(yp[test] > 0.5)
		thisScore = scoreFunc(y[test],y_[test])
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(np.zeros(len(test)),y_[test]))
		meanScore += thisScore
		i += 1

	yf = pd.DataFrame({'loss':y,'rgr':yr,'clf':yp})
	yf.to_csv('base.csv')
	return meanScore/nFolds

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
			#pl.plot(yCV,y_,'o')
			#pl.show()

	meanScore /= nFolds
	return meanScore

def stratClassify(X,y,clf,scoreFunc,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore, i = 0., 1
	allZero = 0.
	yTarget = y.copy()
	yTarget[yTarget > 6] = 7
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)

	for train, test in sss:
		clf.fit(X[train,:],yTarget[train])
		y_ = clf.predict(X[test,:])
		yt = y_.copy().astype('float')
		yt[y_ == 7] *= 2.
		yt[y_ < 7] /= 2.
		thisScore = scoreFunc(y[test],yt)
		thisZero = scoreFunc(y[test],np.zeros(len(test)))
		if verbose:
			"""
			yp = clf.predict_proba(X[test,:])[:,1]
			fpr, tpr, thresh = roc_curve(yTarget[test],yp)
			print "AUC: %f"%auc(fpr,tpr)
			print "Target Samples: %d"%np.sum(yTarget[test])
			"""
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(y[test],np.zeros(len(test))))
			#pl.plot(fpr,tpr,lw=2)
			#pl.show()
			cm = confusion_matrix(yTarget[test],y_)
			print "Confusion Matrix: "
			print cm
		meanScore += thisScore
		allZero += thisZero
		i += 1

	if verbose:
		print "All Zero Average Score: %f"%(allZero/nFolds)
	return meanScore/nFolds

def maxF1(t,p,minval=0,maxval=1,nThresh=4,maxIter=6):
	""""""
	thresh_ = np.linspace(minval,maxval,nThresh)
	maxScore, iter_ = 0, 0
	while iter_ < maxIter:
		fs = [f1_score(t,1*(p > th)) for th in thresh_]
		amax, tmax = np.argmax(fs), np.max(fs)
		if tmax > maxScore:
			maxScore = tmax
		if tmax < maxScore:
			break
		minval, maxval = np.max([0,amax-1]),np.min([nThresh-1,amax+1])
		nThresh *= 2
		thresh_ = np.linspace(thresh_[minval],thresh_[maxval],nThresh)
		iter_ += 1
	print thresh_[amax]
	return maxScore

def maxMAE(t,p,r,minval=0,maxval=1,nThresh=4,maxIter=6):
	""""""
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

def stratF1D(X,y,clf,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore = 0.
	yTarget = 1*((y > 0))
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	iter_ = 0
	numThresh = 512
	f1scores = np.empty((len(sss),numThresh))
	for train, test in sss:
		clf.fit(X[train,:],yTarget[train])
		y_ = clf.decision_function(X[test,:])
		f1scores[iter_,:] = maxF1(yTarget[test],y_,nThresh=numThresh)
		thisScore = np.max(f1scores[iter_,:])
		iter_ += 1
		#fpr, tpr, thresh = roc_curve(yTarget[test],y_)
		#thisScore = auc(fpr,tpr)
		if verbose:
			print "AUC: %f"%thisScore
		meanScore += thisScore

	#print meanScore/nFolds
	flatScore = np.mean(f1scores,axis=0)
	if verbose:
		print "Best Score %f at Best Thresh: %f"%(np.max(flatScore),np.linspace(-1,1.,numThresh)[np.argmax(flatScore)])
	return np.max(flatScore)



def stratF1(X,y,clf,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore = 0.
	yTarget = 1*((y > 0))
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	iter_ = 0
	numThresh = 512
	f1scores = np.empty((len(sss),numThresh))
	for train, test in sss:
		clf.fit(X[train,:],yTarget[train])
		y_ = clf.predict_proba(X[test,:])[:,1]
		f1scores[iter_,:] = maxF1(yTarget[test],y_,nThresh=numThresh)
		thisScore = np.max(f1scores[iter_,:])
		iter_ += 1
		#fpr, tpr, thresh = roc_curve(yTarget[test],y_)
		#thisScore = auc(fpr,tpr)
		if verbose:
			print "AUC: %f"%thisScore
		meanScore += thisScore

	#print meanScore/nFolds
	flatScore = np.mean(f1scores,axis=0)
	if verbose:
		print "Best Score %f at Best Thresh: %f"%(np.max(flatScore),np.linspace(-1,1.,numThresh)[np.argmax(flatScore)])
	return np.max(flatScore)

def stratKFoldF1(X,y,clf,frac=1.,baseAUC=None,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y > 0,n_folds=nFolds)
	yp, y_ = np.empty(y.size), np.empty(y.size)
	np.random.seed(seed)
	for train, test in kf:
		i0, i1 = train[y[train] == 0], train[y[train] > 0]
		numTrain = int(frac*len(i0))
		np.random.shuffle(i0)
		ind_ = np.concatenate((i0[:numTrain],i1))
		clf.fit(X[ind_,:],y[ind_] > 0)
		yp[test] = clf.decision_function(X[test,:])
	
	fpr, tpr, thresh = roc_curve(y>0,yp)
	if baseAUC:
		if np.abs(baseAUC - auc(fpr,tpr)) > 0.01:
			return 0
	return maxF1(y > 0,yp,minval=yp.min(),maxval=yp.max(),nThresh=6)

def stratKFoldAUC(X,y,clf,frac=1.,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y > 0,n_folds=nFolds)
	yp, y_ = np.empty(y.size), np.empty(y.size)
	np.random.seed(seed)
	for train, test in kf:
		i0, i1 = train[y[train] == 0], train[y[train] > 0]
		np.random.shuffle(i0)
		numTrain = int(frac*len(i0))
		ind_ = np.concatenate((i0[:numTrain],i1))
		clf.fit(X[ind_,:],y[ind_] > 3)
		yp[test] = clf.decision_function(X[test,:])
	
	fpr, tpr, thresh = roc_curve(y>0,yp)
	return auc(fpr,tpr)

def stratAUC(X,y,clf,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore = 0.
	yTarget = 1*((y > 0))
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

def stratAUCD(X,y,clf,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore = 0.
	yTarget = 1*((y > 0))
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(X[train,:],yTarget[train])
		y_ = clf.decision_function(X[test,:])
		fpr, tpr, thresh = roc_curve(yTarget[test],y_)
		thisScore = auc(fpr,tpr)
		if verbose:
			print "AUC: %f"%thisScore
		meanScore += thisScore

	return meanScore/nFolds

def greedyReductionAUC(X,y,clf,fnames):
	""""""
	base = [170, 59, 405, 432, 293, 68, 19, 549, 626, 45, 422, 246, 552, 133, 6, 87, 182, 183, 364, 401, 478, 481, 545, 553, 634]
	scoreHist = []
	goodFeats = set(base)
	maxScore = stratAUC(X,y,clf,nFolds=5,verbose=False)
	print "Starting score: %f"%maxScore
	while len(scoreHist) < 2 or scoreHist[-1][0] > scoreHist[-2][0]:
		scores = []
		for iter_ in xrange(X.shape[1]):
			if iter_ not in goodFeats:
				feats = [iter_] + list(goodFeats)
				score = stratAUC(X[:,feats],y,clf,nFolds=5,verbose=False)
				scores.append((score,iter_))
				if score > maxScore:
					print "Feature %s: %f"%(fnames[iter_],score)

		sortScores = sorted(scores)
		goodFeats.add(sortScores[-1][1])
		scoreHist.append(sortScores[-1])
		maxScore = sortScores[-1][0]
		print "Current Features: %s"%sorted(list(goodFeats))
	goodFeats.remove(scoreHist[-1][1])
	goodFeats = sorted(list(goodFeats))
	feats = [fnames[i] for i in goodFeats]
	return feats

def featureScore(x):
	""""""
	X, y, clf = x
	#return holdout(X,y,clf,nFolds=5,verbose=False)
	#return stratF1D(X,y,clf,nFolds=5,verbose=False)
	#return stratAUC(X,y,clf,nFolds=5,verbose=False)
	#return stratKFoldF1(X,y,clf,baseAUC,verbose=False)
	return stratKFoldAUC(X,y,clf,verbose=False)
	r_ = stratKFoldR(X,y,clf,frac=0.1,nFolds=5)
	return mean_absolute_error(y[y > 0],r_[y>0])

def MultiGreedyMAE(X,y,clf,fnames):
	pool = Pool(processes=4)

	#bestFeats = [201, 767, 1, 268, 271, 520, 521] 
	bestFeats = []
	lastScore = 10
	allFeats = [f for f in xrange(X.shape[1])]
	#baseAUC = stratKFoldAUC(X[:,bestFeats],y,clf)
	while True:
		testFeatSets = [[f] + bestFeats for f in allFeats if f not in bestFeats]
		args = [(X[:,fSet],y,clf) for fSet in testFeatSets]
		scores = pool.map(featureScore,args)
		(score, featureSet) = min(zip(scores,testFeatSets))
		print featureSet
		#baseAUC = stratKFoldAUC(X[:,featureSet],y,clf)
		print "Max AUC: %f"%score
		if score > lastScore:
			break
		lastScore = score
		bestFeats = featureSet

	pool.close()
	return [fnames[i] for i in bestFeats]

def MultiGreedyAUC(X,y,clf,fnames):
	pool = Pool(processes=4)

	#bestFeats = [201, 767, 1, 268, 271, 520, 521] 
	bestFeats = [520,521]
	lastScore = 0
	allFeats = [f for f in xrange(X.shape[1])]
	while True:
		testFeatSets = [[f] + bestFeats for f in allFeats if f not in bestFeats]
		args = [(X[:,fSet],y,clf) for fSet in testFeatSets]
		scores = pool.map(featureScore,args)
		(score, featureSet) = max(zip(scores,testFeatSets))
		print featureSet
		#baseAUC = stratKFoldAUC(X[:,featureSet],y,clf)
		print "Max AUC: %f"%score
		if score < lastScore:
			break
		lastScore = score
		bestFeats = featureSet

	pool.close()
	return [fnames[i] for i in bestFeats]