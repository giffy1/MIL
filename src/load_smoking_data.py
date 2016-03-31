# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:20:03 2016

@author: snoran

Script for loading RisQ smoking data by participant.

"""

from __future__ import division
from scipy.stats import kurtosis, skew
import os
from numpy import hstack, vstack, mean, var, sqrt, genfromtxt
from argparse import ArgumentParser
import pickle
import warnings
import csv

import os
import arff
import numpy as np

def load_data(data_dir = 'smoking-data'):
	'''
	Load RisQ dataset.
	
	@param data_dir : 	The directory where the dataset is located. The default is 
				'smoking-data/'.
	
	'''
		
# -----------------------------------------------------------------------------------
#
#							Load Data
#
# -----------------------------------------------------------------------------------
	
	arff_prefix = 'FF_'
	
	X = [] #entire data matrix over all participants (len(X) = # of participants)
	Y = []
	session_start = []
	session_labels = []
	
	for folder in os.listdir(data_dir):
		try:
			int(folder)
		except:
			continue
		
		session_start_i = []
		session_labels_i = []
		
		session_dir = os.path.join(data_dir, folder, 'session')
		
		first_iteration = 1
		for filename in os.listdir(session_dir):
			if not filename.startswith(arff_prefix):
				continue
			
			fullpath = os.path.join(session_dir, filename)
			with open(fullpath, 'rb') as f:
				content = f.read()
			try:
				bug_text = 'class{E,S,D,O,N}'
				bug_index = content.index(bug_text)
				replace = content[:bug_index] + 'class {E,S,D,O,N}' + content[(bug_index + len(bug_text)):]
				
				with open(fullpath, 'wb') as f:
					f.write(replace)
			except ValueError: #substring not found (no bug left)
				pass
			
			dataset = arff.load(open(fullpath, 'rb'))
			if len(dataset['data']) > 0:
				x = np.array(dataset['data'])[:,:-1].astype(float)
				y = 2*(np.array(dataset['data'])[:,-1] == 'S').astype(int)-1
				if first_iteration:
					session_start_i.append(0)
					session_labels_i.append(max(y))
					X_i = x
					Y_i = y
				else:
					session_start_i.append(len(Y_i))
					session_labels_i.append(max(y))
					X_i = np.vstack((X_i, x))
					Y_i = np.hstack((Y_i, y))
					
			first_iteration = 0
			
		X.append(X_i)
		Y.append(Y_i)
		session_start.append(session_start_i)
		session_labels.append(session_labels_i)
	
	#Note: participants 0, 5, 8, 9 and 10 include no smoking sessions
	dataset = {'data': {'X': X, 'Y': Y, 'sessions' : {'start' : session_start, 'labels' : session_labels}}, \
		     'parameters' : {}}

	# TODO: Load/save from pickle?	
	
	return dataset

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--dir", dest="data_dir", default='../data/smoking-data/', \
			help="Directory where the dataset is stored.")	
#	parser.add_argument("--load", dest="load_pickle_path", default='none', \
#			help="Path from which to load the pickle file. This will significantly speed up loading the data. " + \
#			"If 'none' (default), the data will be reloaded from the specified directory.")	
#	parser.add_argument("--save", dest="save_pickle_path", default='none', \
#			help="Path where the data will be saved. If 'none' (default), the data will not be saved.")	
			
	args = parser.parse_args()

	dataset = load_data(**vars(args))
	
	#print dataset