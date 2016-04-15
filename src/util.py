# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:27:08 2016

@author: snoran
"""

from __future__ import division

import numpy as np
from time import time
import datetime
import sys
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from numpy.random import RandomState

def shuffle(seed=None, *args):
	"""
	Shuffles the given lists in parallel.
	"""
	indices = range(len(args[0]))
	prng = RandomState(seed)	
	prng.shuffle(indices)
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
	print(classification_report(y, y_pred))
	print(confusion_matrix(y, y_pred))
	fscore = f1_score(y, y_pred)
	sys.stdout.flush()
	return fscore

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
	
def mil_train_test_split(X_SI, X_B, M):
	"""
	Generates a size kfolds list of train-test splits for cross-validation, 
	specifically for MIL techniques. All bag-level data remains in the 
	training data and a subset of the single-instance bags are used for 
	validation.
	
	@param X_SI : The single instance bags
	@param X_B : Bag-level data
	
	Returns a list of tuples, each of which contains training and test indices. 
	Note : This assumes the final training dataset is X_SI + X_B.
	"""
	n_single_instance_participants = len(X_SI)
	n_bag_participants = len(X_B)
	cv_iterator = []
	if M == 0:
		end_of_test_indices = sum([len(X_SI[k]) for k in range(len(X_SI))])
		end_of_train_indices = end_of_test_indices + sum([len(X_B[k]) for k in range(len(X_B))])
		test_indices = range(end_of_test_indices)
		train_indices = range(end_of_test_indices, end_of_train_indices)
		cv_iterator.append((train_indices, test_indices))
		return cv_iterator
	for k in range(n_single_instance_participants):
		train_indices = []
		index = 0
		
		#instance-level:
		for j in range(n_single_instance_participants):
			if j!=k:
				indices = range(index, index + len(X_SI[j]))
				train_indices.extend(indices[:M])
			else:
				test_indices=range(index, index + len(X_SI[j]))
			index+=len(X_SI[j])
			
		#bag-level:
		for j in range(n_bag_participants):
			indices = range(index, index + len(X_B[j]))
			train_indices.extend(indices)
			index+=len(X_B[j])
		cv_iterator.append((train_indices, test_indices))
	return cv_iterator
