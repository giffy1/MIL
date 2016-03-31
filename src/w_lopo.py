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
from sklearn.svm import SVC
import misvm
from load_eating_data import load_data as load_eating_data
from load_smoking_data import load_data as load_smoking_data
import pickle
from argparse import ArgumentParser
from util import farey, accuracy_precision_recall_fscore, pprint_header, score
from miforest.miforest import MIForest

MIL = {'SIL', 'sMIL', 'sbMIL', 'misvm', 'MIForest'}

def main(dataset, bag_size, active_participant_counter, clf_name, cv, n_iter, cv_method, M, N, K, verbose, \
	   data_dir, load_pickle_path, save_pickle_path, frame_size, step_size, units, \
	   eta_, n_jobs, save_path, description, n_trials, kernel):
	"""
	@param dataset : A string indicating which dataset to load. Choices include 'smoking' and 'eating'
	@param bag_size : The size of the training bags.
	@param active_participant_counter : Index of the held-out participant.
	@param clf_name : Classifier; one of 'SVM', 'LinearSVC', 'RF', 'SIL', 'LinearSIL', 
				'sMIL', 'sbMIL', 'MIForest' or 'misvm'.
	@param cv : Number of cross-validation folds.
	@param n_iter : If the cross-validation search method is 'randomized', then 
				n_iter is the number of randomly sampled parameter tuples.
	@param cv_method : The search method for cross-validation; either 'grid' or 
				'randomized'.
	@param M : The number of labeled training instances.
	@param N : The number of labeled training bags.
	@param K : The number of labeled training instances from the held-out participant.
	@param verbose : Indicates the level of detail to print during run-time.
	@param data_dir : The directory in which the Lab-20 dataset is located.
	@param load_pickle_path : The pickle path where the dataset is stored.
	@param save_pickle_path : The pickle path where the dataset is to be saved.
	@param frame_size : The size of the sliding window used in extracting 
				labeled features over the dataset.
	@param step_size : The stride of the sliding windows used in extracting 
				labeled features over the dataset.
	@param units : The unit of the frame and step size ('u' for samples, 's' for seconds).
	@param eta_ : If the classifier used is sbMIL, eta is the expected density 
				of positive instances in positive bags, between 0.0 and 1.0.
	@param n_jobs : The number of jobs, -1 for full parallelization.
	@param save_path : The path of the file where all results are stored.
	@param description : Description of the evaluation.
	@param n_trials : Number of trials, in case randomness is introduced in each trial.
	@param kernel : If a non-linear SVM-based classifier is used, the kernel 
				can be specified, i.e. 'rbf', 'linear_av', etc.
	"""
	
	if dataset == 'smoking':
		dataset = load_smoking_data(data_dir)
	elif dataset == 'eating':
		dataset = load_eating_data(data_dir, frame_size, step_size, units, load_pickle_path, save_pickle_path)
	X = dataset['data']['X']
	Y = dataset['data']['Y']
	session_start = dataset['data']['sessions']['start']
	session_labels = dataset['data']['sessions']['labels']
	
	if len(X) == 0:
		raise Exception("No dataset loaded: Check to make sure the path is properly set.")
		
						
	#class weights are determined by a Farey sequence to make sure that redundant pairs, 
	#i.e. (1,1) = (2,2), (2,3) = (4,6), etc. are not included.
	class_weights = [{1 : i, -1 : j} for (i,j) in farey(25)[1:]] #ignore first value where i=0
	class_weights.extend([{1 : j, -1 : i} for (i,j) in farey(25)[1:]]) #swap i and j, ignore first value
	C_array = np.logspace(-5, 15, 21, base=2).tolist()
	gamma_array = np.logspace(-15, 3, 19, base=2).tolist()
	eta_array = np.linspace(0,1,9).tolist()
	n_estimators_array = [25,50,75,100,125,150]
	param_grid = {}
	
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
		clf = misvm.sbMIL(kernel=kernel, eta=eta_, C=1.0, verbose=(verbose>1))
	elif clf_name == 'misvm':
		clf = misvm.MISVM(kernel=kernel, C=1.0, verbose=(verbose>1))
	elif clf_name == 'LinearSIL':
		clf = misvm.LinearSIL(C=1.0)
		
	if clf_name in {'RF', 'MIForest'}:
		param_grid.update({'n_estimators' : n_estimators_array})
	
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'RF', 'SVM'}:
		param_grid.update({'class_weight' : class_weights})
		
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'misvm', 'SVM', 'LinearSIL'}:
		param_grid.update({'C' : C_array})
	
	if clf_name in {'SIL', 'sMIL', 'sbMIL', 'misvm', 'SVM'} and kernel == 'rbf':
		param_grid.update({'gamma' : gamma_array})
		
	if clf_name == 'sbMIL':
		param_grid.update({'eta' : eta_array})
	
	data_params = {"Number of Training Bags": N, "Number of Single-Instance Bags" : M, "Test Participant": active_participant_counter}
	cv_params = {"K-Fold": cv, "Method": cv_method, "Parameter Grid" : param_grid, "Number of Iiterations": n_iter}
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
		#select single-instance / bag-level participants at random				
#		np.random.shuffle(train_indices)
		
		si_participant_indices = train_indices[:n_si_participants]
		bag_participant_indices = train_indices[n_si_participants+1:n_si_participants+n_bag_participants+1]
			
		X_SI = np.vstack([X[k] for k in si_participant_indices])
		Y_SI = np.hstack([Y[k] for k in si_participant_indices])

		X_B = np.vstack([X[k] for k in bag_participant_indices])
		Y_B = np.hstack([Y[k] for k in bag_participant_indices])
	
		X_test = X[active_participant_counter]
		Y_test = Y[active_participant_counter]
		
		if N < 0: #TODO: Same for M<0
			N=len(X_B)
			
		if M < 0:
			M=len(X_SI)

# -----------------------------------------------------------------------------------
#
#					   Evaluate Model
#
# -----------------------------------------------------------------------------------
		#convert to bags:
		if clf_name in MIL:
			X_SI = [X_SI[k:k+1, :] for k in xrange(len(X_SI))]
			Y_SI = [max(Y_SI[k:k+1]) for k in xrange(len(Y_SI))]
			
			if bag_size == -1:
				bags = []
				labels = []
				for k in bag_participant_indices:
					for j in range(len(session_labels[k])):
						if session_labels[k][j] > 0:
							if j < len(session_labels[k])-1:
								end = session_start[k][j+1]
							else:
								end = -1 #it's ok to miss 1 sample
							bags.append(X_B[session_start[k][j]:end, :])
							labels.append(1)
				X_B = bags
				Y_B = labels
			else:
				X_B = [X_B[k:k+bag_size, :] for k in xrange(0, len(X_B), bag_size)]
				Y_B = [max(Y_B[k:k+bag_size]) for k in xrange(0, len(Y_B), bag_size)]
			
			X_test = [X_test[k:k+1, :] for k in xrange(len(X_test))]
	
		#shuffle single-instance bags:	
		indices = range(len(X_SI))
		np.random.shuffle(indices)
		X_SI = [X_SI[k] for k in indices[:M]]
		Y_SI = [Y_SI[k] for k in indices[:M]]
		
		#shuffle larger bags:	
		indices = range(len(X_B))
		np.random.shuffle(indices)
		X_B = [X_B[k] for k in indices[:N]]
		Y_B = [Y_B[k] for k in indices[:N]]
		
		#shuffle test data:
		indices = range(len(X_test))
		np.random.shuffle(indices)
		X_test = [X_test[k] for k in indices]
		Y_test = [Y_test[k] for k in indices]
		
		X_T = X_test[:K]
		Y_T = Y_test[:K]
		
		X_test = X_test[K:]
		Y_test = Y_test[K:]
		
		#combine into single training data set with mixed bags and single-instances
		X_train = X_SI + X_B + X_T
		Y_train = Y_SI + Y_B + Y_T
		
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
	
	print("Saving results to %s ..." %save_path)
	
	evaluation = {"Parameters" : params, "Results" : results}
	with open(save_path, 'wb') as f:
		pickle.dump(evaluation, f)
		
if __name__ == "__main__":
	t0=time()
	
	parser = ArgumentParser()
	parser.add_argument("--dataset", dest="dataset", default='eating', type=str, \
			help="Dataset ('eating' or 'smoking')")
	parser.add_argument("--clf", dest="clf_name", default='sbMIL', type=str, \
			help="Classifier ('RF', 'SVM', 'LinearSVC', 'SIL', 'LinearSIL', 'MIForest', 'sMIL', 'sbMIL', 'misvm')")
#	parser.add_argument("--param-grid", dest="param_grid", default=param_grid, \
#			help="Set of hyperparameters learned by cross-validation")
	parser.add_argument("--cv", dest="cv", default=3, type=int, \
			help="Determines split for cross-validation (see GridSearchCV.cv)")
	parser.add_argument("--niter", dest="n_iter", default=10, type=int, \
			help="The number of iterations in randomized cross-validation (see RandomizedSearchCV.cv)")
	parser.add_argument("--cv-method", dest="cv_method", default='randomized', type=str, \
			help="Determines how hyperparameters are learned ('grid' or 'randomized')")
	parser.add_argument("--N", dest="N", default=20, type=int, \
			help="Number of instances used for training in each LOPO iteration")
	parser.add_argument("--M", dest="M", default=20, type=int, \
			help="Number of single-instance bags used for training in each LOPO iteration")
	parser.add_argument("--K", dest="K", default=0, type=int, \
			help="Number of single-instance bags in the training data from the held-out participant.")
	parser.add_argument("--verbose", dest="verbose", default=1, type=int, \
			help="Indicates how much information should be reported (0=None, 1=Some, 2=Quite a bit)")
	parser.add_argument("--bag-size", dest="bag_size", default=-1, type=int, \
			help="If clf is an MIL classifier, bag-size specifies the size of each training bag")
	parser.add_argument("--test-participant", dest="active_participant_counter", default = 0, type=int, \
			help="Index of the held-out participant. The classifier will be evaluated on this individual's data.")
			
	parser.add_argument("--dir", dest="data_dir", default='../data/eating_detection_inertial_ubicomp2015', type=str, \
			help="Directory where the dataset is stored")
	parser.add_argument("--load", dest="load_pickle_path", default='../data/eatng_detection_inertial_ubicomp2015/data.pickle', type=str, \
			help="Path from which to load the pickle file. This will significantly speed up loading the data. If 'none' (default), the data will be reloaded from --dir")	
	parser.add_argument("--save", dest="save_pickle_path", default='../data/eating_detection_inertial_ubicomp2015/data.pickle', type=str, \
			help="Path of the pickle file containing the data. If none (default), the data will not be pickled")	
	parser.add_argument("--save_path", dest="save_path", default='results.pickle', type=str, \
			help="Path of the pickle file containing the data. If none (default), the data will not be pickled")	
	
	#Data parameters for Ubicomp eating dataset
	parser.add_argument("--frame-size", dest="frame_size", default=144, type=int, \
			help="The size of the sliding frame over which instance feature-label pairs are defined")
	parser.add_argument("--step-size", dest="step_size", default=72, type=int, \
			help="The step size of the sliding window")
	parser.add_argument("--units", dest="units", default='u', type=str, \
			help="The units in which the frame size and step size are defined. ('s' for seconds, 'u' for samples)")	
	parser.add_argument("--eta", dest="eta_", default=0.5, type=float, \
			help="Balancing parameter for sbMIL, between 0 and 1 inclusively")	
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, \
			help="Number of threads used (default = 1). Use -1 for maximal parallelization")
	parser.add_argument("--desc", dest="description", default='MIForest test.', type=str, \
			help="Description of the evaluation and parameter selection")
	parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, \
			help="Number of trials over which to average the performance metrics")
	parser.add_argument("--kernel", dest="kernel", default='linear', type=str, \
			help="Kernel type, i.e. 'linear', 'rbf', 'linear_av', etc.")
			
	args = parser.parse_args()
	
	main(**vars(args))
	
	tf=time()
	print("Total time elapsed: %0.2f seconds." %(tf - t0))