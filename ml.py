import numpy as np
import pandas as pd
import pylab as pl

from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

from multiprocessing import Pool

INIT_PARAMS = {
	"LogisticRegression": { 
		'penalty':'l2', 'dual':True, 'tol':0.0001, 'C':1.0, 
		'fit_intercept':True, 'intercept_scaling':1.0, 
        'class_weight':None, 'random_state':1337
        },
    "RandomForestClassifier": {
    	'max_depth':128,'min_samples_split':100,'min_samples_leaf':100,
		'n_jobs':1,'verbose':True,'n_estimators':50,'random_state':1337
		},
	"SGDClassifier":{
		'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':20,
		'shuffle':True,'random_state':1337,'class_weight':None
		},
	"GradientBoostingClassifier":{'max_depth':8, 'subsample':0.5,
		'verbose':2, 'random_state':1337,
		'min_samples_split':100, 'min_samples_leaf':100, 'max_features':10,
		'n_estimators': 125, 'learning_rate': 0.1}
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

def stratHoldout(X,y,clf,scoreFunc,scaleFactor=0.62,decisionThreshold=0.6,
	nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore, i = 0., 1
	allZero = 0.
	yTarget = 1*(y > 0)
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(X[train,:],y[train],thresh=decisionThreshold)
		yp = clf.predict_proba(X[test,:])[:,1]
		yr = clf.predict(X[test,:])

		y_ = yr*(yp > decisionThreshold)*scaleFactor
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

def stratHoldoutMix(X,Z,y,clf,scoreFunc,scaleFactor=0.6,decisionThreshold=0.4,
	nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore, i = 0., 1
	allZero = 0.
	yTarget = 1*(y > 0)
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(X[train,:],y[train],thresh=decisionThreshold)
		yp = clf.predict_proba(X[test,:])[:,1]
		yr = clf.predict(X[test,:])
		
		clf.fit(Z[train,:],y[train],thresh=0.4,cutoff=40)
		zp = clf.predict_proba(Z[test,:])[:,1]
		zr = clf.predict(Z[test,:])
		y_ = yr*(yp > decisionThreshold)*scaleFactor
		y_[(zp > 0.7) & (zr > 40)] = zr[(zp > 0.7) & (zr > 40)]
		y_[y_ < 0] = 0.
		thisScore = scoreFunc(y[test],y_)
		thisZero = scoreFunc(y[test],np.zeros(len(test)))
		if verbose:
			yp = clf.predict_proba(Z[test,:])[:,1]
			fpr, tpr, thresh = roc_curve(y[test]>30,zp)
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


def stratKFold(X,y,clf,scoreFunc,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y,n_folds=nFolds)
	yp, yr, y_ = np.empty(y.size), np.empty(y.size), np.empty(y.size)
	meanScore, i = 0., 1
	for train, test in kf:
		clf.fit(X[train,:],y[train])
		yp[test] = clf.predict_proba(X[test,:])[:,1]
		yr[test] = clf.predict(X[test,:])
		y_[test] = 0.6*yr[test]*(yp[test] > 0.5)
		thisScore = scoreFunc(y[test],y_[test])
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(np.zeros(len(test)),y_[test]))
		meanScore += thisScore
		i += 1

	yf = pd.DataFrame({'loss':y,'rgr':yr,'clf':yp})
	yf.to_csv('base.csv')
	return meanScore/nFolds

def stratKFoldMix(X,Z,y,clf,scoreFunc,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y,n_folds=nFolds)
	yp, yr, y_ = np.empty(y.size), np.empty(y.size), np.empty(y.size)
	zp, zr = np.empty(y.size), np.empty(y.size)
	meanScore, i = 0., 1
	for train, test in kf:
		clf.fit(X[train,:],y[train])
		yp[test] = clf.predict_proba(X[test,:])[:,1]
		yr[test] = clf.predict(X[test,:])

		clf.fit(Z[train,:],y[train],cutoff=40)
		zp[test] = clf.predict_proba(Z[test,:])[:,1]
		zr[test] = clf.predict(Z[test,:])

		y_[test] = 0.6*yr[test]*(yp[test] > 0.5)
		thisScore = scoreFunc(y[test],y_[test])
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(np.zeros(len(test)),y_[test]))
		meanScore += thisScore
		i += 1

	yf = pd.DataFrame({'loss':y,'rgr':yr,'clf':yp,'rgr40':zr,'clf40':zp})
	yf.to_csv('base4.csv')
	return meanScore/nFolds



def stratKFoldMany(X,y,clf,scoreFunc,nFolds=5,seed=1337,verbose=True):
	""""""
	kf = StratifiedKFold(y,n_folds=nFolds)
	yp, yr, y_ = np.empty(y.size), np.empty(y.size), np.empty(y.size)
	y10, y20, y40 = np.empty(y.size), np.empty(y.size), np.empty(y.size)
	r10, r20, r40 = np.empty(y.size), np.empty(y.size), np.empty(y.size)
	y10p = np.empty(y.size)
	meanScore, i = 0., 1
	for train, test in kf:
		clf.fit(X[train,:],y[train],thresh=0.406667)
		yp[test] = clf.predict_proba(X[test,:])[:,1]
		yr[test] = clf.predict(X[test,:])

		# Do the rest
		clf.fit(X[train,:],y[train],thresh=0.406667,cutoff=10)
		y10[test] = clf.predict_proba(X[test,:])[:,1]
		r10[test] = clf.predict(X[test,:])
		clf.clf.fit(X[train,:],(y[train] > 10) & (y[train] <= 20))
		y10p[test] = clf.predict_proba(X[test,:])[:,1]

		clf.fit(X[train,:],y[train],thresh=0.406667,cutoff=20)
		y20[test] = clf.predict_proba(X[test,:])[:,1]
		r20[test] = clf.predict(X[test,:])

		clf.fit(X[train,:],y[train],thresh=0.406667,cutoff=60)
		y40[test] = clf.predict_proba(X[test,:])[:,1]
		r40[test] = clf.predict(X[test,:])

		y_[test] = 0.6*yr[test]*(yp[test] > 0.5)
		thisScore = scoreFunc(y[test],y_[test])
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(np.zeros(len(test)),y_[test]))
		meanScore += thisScore
		i += 1

	yf = pd.DataFrame({'loss':y,'rgr':yr,'clf':yp,
		'clf10':y10,'clf20':y20,'clf40':y40, 'clf10p':y10p,
		'rgr10':r10,'rgr20':r20,'rgr40':r40})
	yf.to_csv('base3.csv')
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
		thisScore = mean_absolute_error(yCV,y_)
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

def stratAUC(X,y,clf,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore = 0.
	yTarget = 1*((y > 40))
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
	return stratAUC(X,y,clf,nFolds=5,verbose=False)

def MultiGreedyAUC(X,y,clf,fnames):
	pool = Pool(processes=4)

	bestFeats = [246,552,133,6, 87, 182, 183, 364, 401, 478, 481, 545, 553, 634] 
	lastScore = 0
	allFeats = [f for f in xrange(X.shape[1])]
	while True:
		testFeatSets = [[f] + bestFeats for f in allFeats if f not in bestFeats]
		args = [(X[:,fSet],y,clf) for fSet in testFeatSets]
		scores = pool.map(featureScore,args)
		(score, featureSet) = max(zip(scores,testFeatSets))
		print featureSet
		print "Max AUC: %f"%score
		if score <= lastScore:
			break
		lastScore = score
		bestFeats = featureSet

	pool.close()
	return [fnames[i] for i in bestFeatures]

