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

import numpy as np
from sklearn.metrics import confusion_matrix
from time import time
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import misvm
import pickle
from argparse import ArgumentParser
from util import farey, accuracy_precision_recall_fscore, pprint_header, score, shuffle
from miforest.miforest import MIForest
import sys

MIL = {'SIL', 'sMIL', 'sbMIL', 'misvm', 'MIForest'}

def single_instances_to_sessions(X, Y, session_labels, session_start, participant_indices):
	"""
	bags the data for the specified participants by sessions. The starting index of 
	each session is defined in session_start and the corresponding labels are defined 
	in session_labels. (X,Y) is the entire dataset.
	"""
	bags = []
	labels = []
	single_instance_labels = []
	for k in participant_indices:
		for j in range(len(session_labels[k])):
			if j < len(session_labels[k])-1:
				end = session_start[k][j+1]
			else:
				end = len(X[k])
			bags.append(X[k][session_start[k][j]:end, :])
			single_instance_labels.append(Y[k][session_start[k][j]:end])
			labels.append(session_labels[k][j])
			
	return bags, labels, single_instance_labels

def main(data_dir, active_participant_counter, bag_size, held_out_bag_size, test_bag_size, M, N, K, \
	   clf_name, eta, kernel, cv_method, cv, n_iter, n_jobs, n_trials, verbose, save, description):
	"""
	@param data_dir : The directory in which the data is located. The directory should contain a 
				load_data.py script with a load_data() method, which returns the feature 
				representation of the dataset.
	@param active_participant_counter : Index of the held-out test participant.
	
	@param bag_size : The size of the training bags. Use -1 for sessions.
	@param held_out_bag_size : The size of the training bags from the held-out participant. Use -1 for sessions.
	@param test_bag_size : The size of the test bags. Use -1 for sessions.
	
	@param M : The number of labeled training instances.
	@param N : The number of labeled training bags.
	@param K : The number of labeled training instances from the held-out participant.
	
	@param clf_name : Classifier; one of 'SVM', 'LinearSVC', 'RF', 'SIL', 'LinearSIL', 
				'sMIL', 'sbMIL', 'MIForest' or 'misvm'.
	@param eta_ : If the classifier used is sbMIL, eta is the expected density 
				of positive instances in positive bags, between 0.0 and 1.0.
	@param kernel : If a non-linear SVM-based classifier is used, the kernel 
				can be specified, i.e. 'rbf', 'linear_av', etc.

	@param cv_method : The search method for cross-validation; either 'grid' or 
				'randomized'.			
	@param cv : Number of cross-validation folds.
	@param n_iter : If the cross-validation search method is 'randomized', then 
				n_iter is the number of randomly sampled parameter tuples.
	@param n_jobs : The number of jobs, -1 for full parallelization.
	@param n_trials : Number of trials, in case randomness is introduced in each trial.

	@param verbose : Indicates the level of detail to be displayed during run-time.
	@param save : The path of the file where results are stored.
	@param description : Description of the evaluation to be saved with the results.

	"""
	
	sys.path.insert(0, data_dir)
	from load_data import load_data
	
	dataset = load_data(data_dir)
	X = dataset['data']['X']
	Y = dataset['data']['Y']
	session_start = dataset['data']['sessions']['start']
	session_labels = dataset['data']['sessions']['labels']	
	print data_dir
	print dataset['description']
						
	
	if clf_name == 'RF':
		clf = RandomForestClassifier(n_estimators=185, verbose=(verbose>1))
	elif clf_name == 'SVM':
		clf = SVC(kernel=kernel, verbose=(verbose>1))
	elif clf_name == 'SIL':
		clf = misvm.SIL(kernel=kernel, C=1.0, verbose=(verbose>1))
	elif clf_name == 'MIForest':
		clf = MIForest(n_estimators=50, directory="miforest",  prefix="eating")
	elif clf_name == 'sMIL':
		clf = misvm.sMIL(kernel=kernel, C=1.0, verbose=(verbose>1))
	elif clf_name == 'sbMIL':
		clf = misvm.sbMIL(kernel=kernel, eta=eta, C=1.0, verbose=(verbose>1))
	elif clf_name == 'misvm':
		clf = misvm.MISVM(kernel=kernel, C=1.0, verbose=(verbose>1))
	elif clf_name == 'LinearSIL':
		clf = misvm.LinearSIL(C=1.0)
	elif clf_name == 'LinearSVC':
		clf = LinearSVC(C=1.0)
		
	#class weights are determined by a Farey sequence to make sure that redundant pairs, 
	#i.e. (1,1) = (2,2), (2,3) = (4,6), etc. are not included.
	class_weights = [{1 : i, -1 : j} for (i,j) in farey(25)[1:]] #ignore first value where i=0
	class_weights.extend([{1 : j, -1 : i} for (i,j) in farey(25)[1:]]) #swap i and j, ignore first value
	C_array = np.logspace(-5, 15, 21, base=2).tolist()
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
	
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'misvm', 'SVM'} and kernel == 'rbf':
		param_grid.update({'gamma' : gamma_array})
		
	if clf_name == 'sbMIL':
		param_grid.update({'eta' : eta_array})
	
	data_params = {"Number of Training Bags": N, "Number of Single-Instance Bags" : M, "Test Participant": active_participant_counter}
	cv_params = {"K-Fold": cv, "Method": cv_method, "Parameter Grid" : param_grid, "Number of Iterations": n_iter}
	params = {
		"Bag Size": bag_size, \
		"Data": data_params, \
		"Classifier": str(clf), \
		"Number of Trials": n_trials, \
		"CV": cv_params \
	}	
	results = {
		"Confusion Matrix" : {"Training" : np.zeros((2,2)), "Test" : np.zeros((2,2))}, \
		"Precision": {"Training" : 0.0, "Test" : 0.0}, \
		"Recall": {"Training" : 0.0, "Test" : 0.0}, \
		"F1 Score": {"Training" : 0.0, "Test" : 0.0, "Validation" : 0.0} \
	}
		
	participant_indices = range(len(X))
	n_si_participants = 5
	n_bag_participants = len(X) - n_si_participants - 1
	
	if verbose:
		pprint_header("Train Model for Participant: " + str(active_participant_counter + (active_participant_counter>=13) + 1))
	
	for T in xrange(1,n_trials+1): #allow multiple trials to account for randomness
		pprint_header("Trial: " + str(T))
		
		#indices for participants in training data; skip active participant counter:
		train_indices = participant_indices[:active_participant_counter] + participant_indices[active_participant_counter+1:]
		
		si_participant_indices = train_indices[:n_si_participants]
		bag_participant_indices = train_indices[n_si_participants+1:n_si_participants+n_bag_participants+1]
		
		#single-instance training data:
		X_SI = np.vstack([X[k] for k in si_participant_indices])
		Y_SI = np.hstack([Y[k] for k in si_participant_indices])

		#bag-level training data:
		X_B = np.vstack([X[k] for k in bag_participant_indices])
		Y_B = np.hstack([Y[k] for k in bag_participant_indices])
	
		#test data
		X_test = X[active_participant_counter]
		Y_test = Y[active_participant_counter]

		#convert to bags:
		if clf_name in MIL:
			X_SI = [X_SI[k:k+1, :] for k in xrange(len(X_SI))]
			Y_SI = [max(Y_SI[k:k+1]) for k in xrange(len(Y_SI))]
			
			if bag_size == -1:
				X_B, Y_B, _ = single_instances_to_sessions(X, Y, session_labels, session_start, bag_participant_indices)
			else:
				X_B = [X_B[k:k+bag_size, :] for k in xrange(0, len(X_B), bag_size)]
				Y_B = [max(Y_B[k:k+bag_size]) for k in xrange(0, len(Y_B), bag_size)]
			
			if held_out_bag_size == -1:
				X_T, Y_T, Y_si = single_instances_to_sessions(X, Y, session_labels, session_start, [active_participant_counter])
			else:
				X_T = [X_test[k:k+bag_size, :] for k in xrange(0,len(X_test), held_out_bag_size)]
				Y_si = [Y_test[k:k+bag_size] for k in xrange(0,len(Y_test), held_out_bag_size)]
				Y_T = [max(y_t) for y_t in Y_si]
								
			X_T, Y_T, Y_si = shuffle(X_T, Y_T, Y_si)	

			# convert remaining bags back to test instances
			X_test = []
			Y_test = []
			for i, (x_t, y_si) in enumerate(zip(X_T, Y_si)[K:]):
				for (x,y) in zip(x_t, y_si):
					X_test.append(x)
					Y_test.append(y)
					
			X_test = [np.asarray(X_test)[k:k+test_bag_size, :] for k in xrange(0, len(X_test), test_bag_size)]
			Y_test = [max(Y_test[k:k+test_bag_size]) for k in xrange(0, len(Y_test), test_bag_size)]

		else: # standard supervised learning case
			X_T = X_test[:K]
			X_T = X_test[:K]
			X_test = X_test[K:]
			Y_test = Y_test[K:]
			
		if N < 0:
			N=len(X_B)
			
		if M < 0:
			M=len(X_SI)
			
		X_SI, Y_SI = shuffle(X_SI, Y_SI)
		X_B, Y_B = shuffle(X_B, Y_B)
		X_test, Y_test = shuffle(X_test, Y_test)

		#combine into single training data set with mixed bags and single-instances
		X_train = []
		Y_train = []
		if M > 0:
			X_train += X_SI[:M]
			Y_train += Y_SI[:M]
		if N > 0:
			X_train += X_B[:N]
			Y_train += Y_B[:N]
		if K > 0:
			X_train += X_T[:K]
			Y_train += Y_T[:K]
		
		if clf_name in MIL:
			print ("Total number of bags : %d" %len(X_train))
			print ("Feature Dimensionality: %d " %X_train[0].shape[1])
		else:
			print ("Total number of instances : %d" %len(X_train))
			print("Feature Dimensionality %d " %len(X_train[0]))
		
		sys.stdout.flush()
		if cv_method == 'grid':
			gs = GridSearchCV(clf, param_grid, scoring=score, cv=cv, verbose=verbose, n_jobs = n_jobs)
		elif cv_method == 'randomized':
			#scoring='f1_weighted'
			gs = RandomizedSearchCV(clf, param_distributions=param_grid, scoring=score, cv=cv, n_jobs = n_jobs, n_iter=n_iter, verbose=verbose)
		
		t0 = time()
		gs = gs.fit(X_train, Y_train)
		tf = time()

		print("Time elapsed: %0.2f seconds." %(tf-t0))		
		
		print("Best params: ")
		print(gs.best_params_)	
		print("Best F1-score on training data: %0.2f%%" %(100*gs.best_score_))
		results['F1 Score']['Validation'] += gs.best_score_
		
		if clf_name == 'MIForest': #for MIForest, we need to pass in Y as well
			#check training accuracy to start:
			y_pred = 2*np.greater(gs.best_estimator_.predict(X_train, Y_train),0)-1	
		else: #for MIForest, we need to pass in Y as well
			#check training accuracy to start:
			y_pred = 2*np.greater(gs.best_estimator_.predict(X_train),0)-1
		
		conf = confusion_matrix(Y_train, y_pred, [-1,+1])
		print("Confusion matrix on the training data:")
		print(conf)
		results['Confusion Matrix']['Training'] += conf
		
		if clf_name == 'MIForest':
			y_pred = 2*np.greater(gs.best_estimator_.predict(X_test, Y_test),0)-1
		else:
			y_pred = 2*np.greater(gs.best_estimator_.predict(X_test),0)-1
			
		conf = confusion_matrix(Y_test, y_pred, [-1,+1])
		print("Confusion matrix on the test data:")
		print(conf)
		results['Confusion Matrix']['Test'] += conf
		
	pprint_header("Results")
	
	conf = results['Confusion Matrix']['Training']
	avg_precision, avg_recall, avg_fscore = accuracy_precision_recall_fscore(conf)[1][1]
	results['F1 Score']['Training'] = avg_fscore
	results['Precision']['Training'] = avg_precision
	results['Recall']['Training'] = avg_recall	

	print("Average Precision on the training data: %0.2f%%" %(100*avg_precision))
	print("Average Recall on the training data: %0.2f%%" %(100*avg_recall))
	print("Average F1 Score on the training data: %0.2f%%\n" %(100*avg_fscore))	
	
	conf = results['Confusion Matrix']['Test']
	avg_precision, avg_recall, avg_fscore = accuracy_precision_recall_fscore(conf)[1][1]
	results['F1 Score']['Test'] = avg_fscore
	results['Precision']['Test'] = avg_precision
	results['Recall']['Test'] = avg_recall	
	
	print("Average Precision on the test data: %0.2f%%" %(100*avg_precision))
	print("Average Recall on the test data: %0.2f%%" %(100*avg_recall))
	print("Average F1 Score on the test data: %0.2f%%\n" %(100*avg_fscore))
	
	if save != 'none':
		print("Saving results to %s ..." %save)
		
		evaluation = {"Parameters" : params, "Results" : results}
		with open(save, 'wb') as f:
			pickle.dump(evaluation, f)
		
if __name__ == "__main__":
	t0=time()
	
	parser = ArgumentParser()
	
	parser.add_argument("--data-dir", dest="data_dir", default='../data/eating_detection_inertial_ubicomp2015', type=str, \
			help="Directory where the dataset is stored")
	parser.add_argument("--test-participant", dest="active_participant_counter", default = 0, type=int, \
			help="Index of the held-out participant. The classifier will be evaluated on this individual's data.")
	
	parser.add_argument("--bag-size", dest="bag_size", default=-1, type=int, \
			help="If clf is an MIL classifier, bag-size specifies the size of each training bag")
	parser.add_argument("--held-out-bag-size", dest="held_out_bag_size", default=-1, type=int, \
			help=".")
	parser.add_argument("--test-bag-size", dest="test_bag_size", default=1, type=int, \
			help=".")	

	parser.add_argument("--N", dest="N", default=20, type=int, \
			help="Number of instances used for training in each LOPO iteration")
	parser.add_argument("--M", dest="M", default=100, type=int, \
			help="Number of single-instance bags used for training in each LOPO iteration")
	parser.add_argument("--K", dest="K", default=0, type=int, \
			help="Number of single-instance bags in the training data from the held-out participant.")
	
	parser.add_argument("--clf", dest="clf_name", default='sbMIL', type=str, \
			help="Classifier ('RF', 'SVM', 'LinearSVC', 'SIL', 'LinearSIL', 'MIForest', 'sMIL', 'sbMIL', 'misvm')")
	parser.add_argument("--eta", dest="eta", default=0.5, type=float, \
			help="Balancing parameter for sbMIL, between 0 and 1 inclusively")	
	parser.add_argument("--kernel", dest="kernel", default='linear', type=str, \
			help="Kernel type, i.e. 'linear', 'rbf', 'linear_av', etc.")
			
	parser.add_argument("--cv-method", dest="cv_method", default='randomized', type=str, \
			help="Determines how hyperparameters are learned ('grid' or 'randomized')")
	parser.add_argument("--cv", dest="cv", default=3, type=int, \
			help="Determines split for cross-validation (see GridSearchCV.cv)")
	parser.add_argument("--n-iter", dest="n_iter", default=10, type=int, \
			help="The number of iterations in randomized cross-validation (see RandomizedSearchCV.cv)")
	parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, \
			help="Number of trials over which to average the performance metrics")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, \
			help="Number of threads used (default = 1). Use -1 for maximal parallelization")
			
	parser.add_argument("--verbose", dest="verbose", default=1, type=int, \
			help="Indicates how much information should be reported (0=None, 1=Some, 2=Quite a bit)")
	parser.add_argument("--save", dest="save", default='none', type=str, \
			help="Path of the pickle file containing the data. If none (default), the data will not be pickled")	
	parser.add_argument("--desc", dest="description", default='MIForest test.', type=str, \
			help="Description of the evaluation and parameter selection")
			
	args = parser.parse_args()
	
	main(**vars(args))
	
	tf=time()
	print("Total time elapsed: %0.2f seconds." %(tf - t0))