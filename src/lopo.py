# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:22:34 2016

@author: snoran

Leave-one-participant-out evaluation over Lab-20 eating dataset. Evaluation 
can be done using standard supervised learning methods or multi-instance learning. 
If multi-instance learning is used, then the classifier may be trained using 
either or both single-instance bags or larger bags. Additionally, training data 
may include labelled instances or bags from the held-out participant, in order 
to enhance generalizability.
"""

from __future__ import division

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from time import time
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import misvm
import pickle
from argparse import ArgumentParser
from util import farey, accuracy_precision_recall_fscore, score, mil_train_test_split, pprint_header
from miforest.miforest import MIForest
import json

MIL = {'SIL', 'sMIL', 'sbMIL', 'misvm', 'MIForest'}

def get_param_grid_by_clf(clf_name, kernel='linear'):
	"""
	Generates a reasonable parameter grid (i.e. for cross-validation) for the 
	given classifier. For SVM based techniques (sbMIL, SIL, SVC), this includes 
	generating a list of C values and class_weight pairs. 
	"""	
	
	#class weights are determined by a Farey sequence to make sure that redundant pairs, 
	#i.e. (1,1) = (2,2), (2,3) = (4,6), etc. are not included.
	#class_weights = [{1 : i, -1 : j} for (i,j) in farey(20)[1:]] #ignore first value where i=0
	#class_weights = [{1 : j, -1 : i} for (i,j) in farey(10)[1:]] #swap i and j, ignore first value
	class_weights = [{1: 0.8, -1: 0.2}, {1: 0.825, -1: 0.175}, {1: 0.85, -1: 0.15}, {1: 0.875, -1: 0.125}, {1: 0.9, -1: 0.1}, {1: 0.925, -1: 0.075}, {1: 0.95, -1: 0.05}, {1: 0.975, -1: 0.025}, {1: 0.99, -1: 0.01}]
	C_array = np.logspace(5, 16, 12, base=2).tolist()
	gamma_array = np.logspace(-15, 3, 19, base=2).tolist()
	eta_array = np.linspace(0,1,9).tolist()
	n_estimators_array = [25,50,75,100,125,150]
	param_grid = {}
	
	if clf_name in {'RF', 'MIForest'}:
		param_grid.update({'n_estimators' : n_estimators_array})
	
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'RF', 'SVM', 'LinearSVC'}:
		param_grid.update({'class_weight' : class_weights})
		
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'misvm', 'SVM', 'LinearSIL', 'LinearSVC'}:
		param_grid.update({'C' : C_array})
	
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'misvm', 'SVM'} and kernel.startswith('rbf'):
		param_grid.update({'gamma' : gamma_array})
		
	if clf_name == 'sbMIL':
		param_grid.update({'eta' : eta_array})
		
	return param_grid
	
def get_clf_by_name(clf_name, **kwargs):
	"""
	Returns an instance of a classifier given its name and its parameters. 
	The parameters are specified as key-word arguments to the method, following 
	the name of the classifier. Any parameter may be included as long as it 
	is included in the classifier implementation.
	
	@param clf_name The name of the classifier
	@param kwargs key-word arguments including the model parameters
	"""
	
	if clf_name == 'RF':
		clf = RandomForestClassifier(**kwargs)
	elif clf_name == 'SVM':
		clf = SVC(**kwargs)
	elif clf_name == 'SIL':
		clf = misvm.SIL(**kwargs)
	elif clf_name == 'MIForest':
		clf = MIForest(**kwargs)
	elif clf_name == 'sMIL':
		clf = misvm.sMIL(**kwargs)
	elif clf_name == 'sbMIL':
		clf = misvm.sbMIL(**kwargs)
	elif clf_name == 'misvm':
		clf = misvm.MISVM(**kwargs)
	elif clf_name == 'LinearSIL':
		clf = misvm.LinearSIL(**kwargs)
	elif clf_name == 'LinearSVC':
		clf = LinearSVC(**kwargs)
	return clf
	
def parse_clf(clf_str):
	start = clf_str.index('(')
	end = clf_str.index(')')
	param_str = '{' + clf_str[start+1:end] + '}'
	params = json.loads(param_str)
	clf_name = clf_str[:start]

	return clf_name, params

def main(data_file, clf_str, cv_method, n_iter, n_jobs, verbose, save, description):
	"""
	TODO: Doc string
	"""
	
	with open(data_file, 'rb') as f:
		data = pickle.load(f)
	
	X_SI = data['training']['instance']['X']
	Y_SI = data['training']['instance']['Y']
	X_B = data['training']['bag']['X']
	Y_B = data['training']['bag']['Y']
	X_train = []
	Y_train = []
	for p in range(len(X_SI)):
		X_train.extend(X_SI[p])
		Y_train.extend(Y_SI[p])
	n_single_instances = len(X_train)
	
	#for class weights:
#	N1 = np.sum(np.greater(Y_train, 0))
#	N0 = np.sum(np.less(Y_train, 0))
		
	for p in range(len(X_B)):
		X_train.extend(X_B[p])
		Y_train.extend(Y_B[p])
	n_bags = len(X_train) - n_single_instances
	X_test = data['test']['X']
	Y_test = data['test']['Y']
	
	clf_name, clf_params = parse_clf(clf_str)
#	if N0 + N1 == 0:
#		clf_params['class_weight'] = {1 : 0.9, -1 : 0.1}
#	else:
#		clf_params['class_weight'] = {1 : N0/(N0 + N1), -1 : N1/(N0 + N1)}
#	print clf_params['class_weight']
	clf = get_clf_by_name(clf_name, **clf_params)
	param_grid = get_param_grid_by_clf(clf_name, clf_params.get("kernel", "linear"))

	results = {
		"Confusion Matrix" : {"Training" : np.zeros((2,2)), "Test" : np.zeros((2,2))}, \
		"Precision": {"Training" : 0.0, "Test" : 0.0}, \
		"Recall": {"Training" : 0.0, "Test" : 0.0}, \
		"F1 Score": {"Training" : 0.0, "Test" : 0.0, "Validation" : 0.0} \
	}
	
	cv_iterator = mil_train_test_split(X_SI, X_B)
	
	pprint_header("Number of bags : %d    Number of single instances: %d" %(n_bags, n_single_instances))

	if cv_method == 'grid':
		gs = GridSearchCV(clf, param_grid, scoring=score, cv=cv_iterator, verbose=verbose, n_jobs = n_jobs, refit=False)
	elif cv_method == 'randomized':
		gs = RandomizedSearchCV(clf, param_distributions=param_grid, scoring=score, cv=cv_iterator, n_jobs = n_jobs, n_iter=n_iter, verbose=verbose, refit=False)
	
	t0 = time()
	gs = gs.fit(X_train, Y_train)
	tf = time()
	
	print("Best parameters set found on development set:\n")
	print(gs.best_params_)
	print("\nGrid scores on development set:\n")
	for params, mean_score, scores in gs.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r"
		% (mean_score, scores.std() * 2, params))
	
	clf.set_params(**gs.best_params_)
	clf.fit(X_train, Y_train)
	print("\nDetailed classification report:\n")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.\n")
	y_true, y_pred = Y_test, 2*np.greater(clf.predict(X_test),0)-1
	print(classification_report(y_true, y_pred))

	print("\nTime elapsed: %0.2f seconds." %(tf-t0))
	
#	if clf_name == 'MIForest': #for MIForest, we need to pass in Y as well
#		#check training accuracy to start:
#		y_pred = 2*np.greater(gs.best_estimator_.predict(X_train, Y_train),0)-1	
#	else: #for MIForest, we need to pass in Y as well
#		#check training accuracy to start:
#		y_pred = 2*np.greater(gs.best_estimator_.predict(X_train),0)-1
#	
#	conf = confusion_matrix(Y_train, y_pred, [-1,+1])
#	print("Confusion matrix on the training data:")
#	print(conf)
#	results['Confusion Matrix']['Training'] = conf
#	
#	precision, recall, fscore = accuracy_precision_recall_fscore(conf)[1][1]
#	results['F1 Score']['Training'] = fscore
#	results['Precision']['Training'] = precision
#	results['Recall']['Training'] = recall	
	
	
#	if clf_name == 'MIForest':
#		y_pred = 2*np.greater(gs.best_estimator_.predict(X_test, Y_test),0)-1
#	else:
#		y_pred = 2*np.greater(gs.best_estimator_.predict(X_test),0)-1
#		
	conf = confusion_matrix(y_true, y_pred, [-1,+1])
	print("Confusion matrix on the test data:")
	print(conf)
	results['Confusion Matrix']['Test'] = conf
		
	precision, recall, fscore = accuracy_precision_recall_fscore(conf)[1][1]
	results['F1 Score']['Test'] = fscore
	results['Precision']['Test'] = precision
	results['Recall']['Test'] = recall	
	
	print("Precision on the test data: %0.2f%%" %(100*precision))
	print("Recall on the test data: %0.2f%%" %(100*recall))
	print("F1 Score on the test data: %0.2f%%\n" %(100*fscore))
	
	evaluation = {"Description": description, "Results" : results}	
	if save != 'none':
		print("Saving results to %s ..." %save)

		with open(save, 'wb') as f:
			pickle.dump(evaluation, f)
		
	return evaluation
		
if __name__ == "__main__":
	parser = ArgumentParser()
	
	parser.add_argument("-d", "--data", dest="data_file", default='data.pickle', type=str, \
			help="Directory where the dataset is stored")

	parser.add_argument("--clf", dest="clf_str", default='sbMIL("verbose":0)', type=str, \
			help="Classifier ('RF', 'SVM', 'LinearSVC', 'SIL', 'LinearSIL', 'MIForest', 'sMIL', 'sbMIL', 'misvm')")
			
	parser.add_argument("--cv-method", dest="cv_method", default='randomized', type=str, \
			help="Determines how hyperparameters are learned ('grid' or 'randomized')")
	parser.add_argument("--n-iter", dest="n_iter", default=10, type=int, \
			help="The number of iterations in randomized cross-validation (see RandomizedSearchCV.cv)")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, \
			help="Number of threads used (default = 1). Use -1 for maximal parallelization")
			
	parser.add_argument("--verbose", dest="verbose", default=0, type=int, \
			help="Indicates how much information should be reported (0=None, 1=Some, 2=Quite a bit)")
	parser.add_argument("--save", dest="save", default='none', type=str, \
			help="Path of the pickle file containing the data. If none (default), the data will not be pickled")	
	parser.add_argument("--desc", dest="description", default='', type=str, \
			help="Description of the evaluation and parameter selection")
	
	args = parser.parse_args()
	
	results = main(**vars(args))