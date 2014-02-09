import numpy as np
import pandas as pd
import pylab as pl

from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

INIT_PARAMS = {
	"LogisticRegression": { 
		'penalty':'l2', 'dual':True, 'tol':0.0001, 'C':1.0, 
		'fit_intercept':True, 'intercept_scaling':1.0, 
        'class_weight':None, 'random_state':1337
        },
    "RandomForestClassifier": {
    	'max_depth':128,'min_samples_split':100,'min_samples_leaf':100,
		'n_jobs':4,'verbose':True,'n_estimators':50
		},
	"SGDClassifier":{
		'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':20,
		'shuffle':True,'random_state':1337,'class_weight':None
		}
}


class Cascaded(object):
	"""
	"""
	def __init__(self,classifier,regressor):
		self.clf = classifier
		self.rgr = regressor

	def fit(self,X,y):
		""""""
		self.clf.fit(X,(y > 0))
		self.rgr.fit(X[y > 0],y[y > 0])

	def predict(self,X):
		""""""
		# Fit a regressor
		return 0.75*self.rgr.predict(X)*self.clf.predict(X)
	
	def predict_proba(self,X):
		""""""
		return self.clf.predict_proba(X)

def stratHoldout(X,y,clf,scoreFunc,nFolds=10,fraction=0.2,seed=1337,verbose=True):
	""""""
	meanScore, i = 0., 1
	allZero = 0.
	yTarget = 1*(y > 0)
	sss = StratifiedShuffleSplit(yTarget,n_iter=nFolds,test_size=fraction,random_state=seed)
	for train, test in sss:
		clf.fit(X[train,:],y[train])
		y_ = clf.predict(X[test,:])
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
			pl.plot(fpr,tpr,lw=2)
			pl.show()
			#cm = confusion_matrix(yTarget[test],y_)
			#print "Confusion Matrix: "
			#print cm
		meanScore += thisScore
		allZero += thisZero
		i += 1

	if verbose:
		print "All Zero Average Score: %f"%(allZero/nFolds)
	return meanScore/nFolds

def stratKFold(X,y,clf,scoreFunc,nFolds=10,seed=1337,verbose=True):
	""""""
	kf = KFold(y,n_folds=nFolds,shuffle=True,random_state=seed)
	y_ = np.empty(y.size)
	meanScore, i = 0., 1
	for train, test in kf:
		clf.fit(X[train,:],y[train])
		y_[test] = clf.predict(X[test,:])
		thisScore = scoreFunc(y[test],y_[test])
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore,i,nFolds)
			print "All Zeros Score: %f"%(scoreFunc(np.zeros(len(test)),y_))
		meanScore += thisScore
		i += 1

	return meanScore/nFolds, y_
