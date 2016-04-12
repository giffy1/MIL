# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:27:08 2016

@author: snoran
"""

import numpy as np
from time import time
import datetime
from sklearn.metrics import f1_score

def shuffle(*args):
	"""
	Shuffles the given lists in parallel.
	"""
	indices = range(len(args[0]))
	np.random.shuffle(indices)
	shuffled = []
	for i,lst in enumerate(args):
		shuffled.append([lst[k] for k in indices])
	return shuffled

def score(estimator, X, y):
	"""
	Returns the mean F1 Score on the given test data and labels.
	In multi-label classification, this is the subset accuracy
	which is a harsh metric since you require for each sample that
	each label set be correctly predicted.
	Parameters
	----------
	X : array-like, shape = (n_samples, n_features)
		Test samples.
	y : array-like, shape = (n_samples) or (n_samples, n_outputs)
		True labels for X.
	Returns
	-------
	score : float
		Mean F1 score of self.predict(X) wrt. y.
	"""
	y_pred = 2*np.greater(estimator.predict(X), 0) - 1 #+/-1
	return f1_score(y, y_pred)

def farey( n, asc=True ):
	"""
	Returns the nth Farey sequence, either ascending or descending.
	"""
	fs=[]    
	if asc: 
	        a, b, c, d = 0, 1,  1 , n     # (*)
	else:
	        a, b, c, d = 1, 1, n-1, n     # (*)
	fs.append((a,b))
	while (asc and c <= n) or (not asc and a > 0):
		k = int((n + b)/d)
		a, b, c, d = c, d, k*c - a, k*d - b
		fs.append((a,b))
	return fs
	
def accuracy_precision_recall_fscore(conf):
	"""
	Given a confusion matrix, the accuracy, precision, recall and F1 score is computed.
	"""
	accuracy = np.sum(np.diag(conf)) / np.sum(conf) 
	precision = np.diag(conf) / np.sum(conf, axis=0)
	recall = np.diag(conf) / np.sum(conf, axis=1)
	fscore = 2 * precision * recall / (precision + recall)
	return accuracy, zip(precision, recall, fscore)
	
def pprint_header(header):
	"""
	Formats and prints the specified header text 
	"""
	ts = time()
	current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	print ""
	print "---------------------------------------------------------"
	print ""
	print ""
	print ""
	print ""
	print header
	print ""
	print current_time
	print ""
	print ""
	print ""
	print ""
	print "---------------------------------------------------------"
	print ""
	
def mil_train_test_split(X_SI, X_B, kfolds):
	"""
	Generates a size kfolds list of train-test splits for cross-validation, 
	specifically for MIL techniques. All bag-level data remains in the 
	training data and a subset of the single-instance bags are used for 
	validation.
	
	@param X_SI : The single instance bags
	@param X_B : Bag-level data
	@param kfolds : The number of folds
	
	Returns a list of tuples, each of which contains training and test indices. 
	Note : This assumes the final training dataset is X_SI + X_B.
	"""
	m = len(X_SI)
	n = len(X_B)
	cv_iterator = []
	indices, = shuffle(range(m))
	samples_per_fold = int(m / kfolds)
	for k in range(kfolds):
		test_indices = indices[k*samples_per_fold:(k+1)*samples_per_fold]
		train_indices = indices[:k*samples_per_fold] + indices[(k+1)*samples_per_fold:]
		train_indices.extend(range(m, m + n))
		cv_iterator.append((train_indices, test_indices))
	return cv_iterator