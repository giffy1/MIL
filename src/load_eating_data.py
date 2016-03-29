# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:20:03 2016

@author: snoran

Script for loading, pre-processing and feature extraction of
Edison Thomaz's Lab-20 eating dataset.

"""

from __future__ import division
from scipy.stats import kurtosis, skew
import os
from numpy import hstack, vstack, mean, var, sqrt, genfromtxt
from argparse import ArgumentParser
import pickle
import warnings
import csv

def load_data(data_dir = 'eating_detection_inertial_ubicomp2015', frame_size = 144, \
		step_size = 72, units = 'u', load_pickle_path = 'none', save_pickle_path = 'none'):
	'''
	Load Edison's Lab-20 eating dataset.
	
	@param data_dir : 	The directory where the dataset is located. The default is 
				'eating_detection_inertial_ubicomp2015/', which matches the 
				directory Edison uses.
	@param frame_size : The size of the sliding window over which classification 
				is done. This may be in seconds (if units='s') or in 
				samples (if units='u').
	@param step_size : The step size of the sliding window over which classification 
				is done. It is common to choose step_size = frame_size / 2, 
				meaning there is 50% overlap. This may be in seconds (if 
				units='s') or in samples (if units='u').
	@param units : The unit of the provided frame size and step size. This may 
				be either 'u' for samples or 's' for seconds.
	@param load_pickle_path : The path where the .pickle data file is stored. By 
				default, load_pickle_path='none', in which case the data 
				is reloaded. Setting load_pickle_path appropriately will 
				significantly speed up data loading.
	@param save_pickle_path : The path where the .pickle data should be saved. By 
				default, save_pickle_path='none', in which case the data 
				is not saved.
	
	'''
		
	sampling_rate = 25
	eating_labels = {1, 2, 3}
		
	#load data from pickle file if possible (faster):
	if load_pickle_path != 'none':
		if not load_pickle_path.endswith('.pickle'):
			load_pickle_path += '.pickle'	
		
		try:
			with open(load_pickle_path, 'rb') as handle:
				
				dataset = pickle.load(handle) 
				if int(dataset['parameters']['frame_size']) != frame_size:
					warnings.warn('The frame size parameter does not match the input argument.')
				if int(dataset['parameters']['step_size']) != step_size:
					warnings.warn('The step size parameter does not match the input argument.')
				if dataset['parameters']['units'] != units:
					warnings.warn('The units parameter does not match the input argument.')
				
				return dataset
		except:
			print("Failed to load data from " + str(load_pickle_path))
			pass
		
	if units == 's':
		frame_size = frame_size * sampling_rate
		step_size = step_size * sampling_rate
	elif units != 'u':
		raise ValueError("Invalid units" + str(units) + ". Expected either 's' for seconds or 'u' for samples.")

	n_participants = 21
		
# -----------------------------------------------------------------------------------
#
#							Load Data
#
# -----------------------------------------------------------------------------------
	
	X = [] #entire data matrix over all participants (len(X) = # of participants)
	Y = []
	activities_time = []
	activities_eatingflag = []
	
	for participant_counter in xrange(1,n_participants+1,1):
		if participant_counter==14:
			continue #ignore participant 14, invalid data
	
		path = os.path.join(data_dir, 'participants', str(participant_counter), 'datafiles', 'waccel_tc_ss_label.csv')
		print "Loading: " + path
		L_T = genfromtxt(path, delimiter=',')
	
		# Remove the relative timestamp
		L_T = L_T[:,1:]
	
		# Number of inputs
		number_of_inputs = L_T.shape[1]-1
	
		print ""
		print "---------------------------------------------------------"
		print " Computing Features for Participant: " + str(participant_counter)
		print "---------------------------------------------------------"
		print ""
	
		pos_examples_counter = 0
		neg_examples_counter = 0
	
		# Calculate features for frame
		for counter in xrange(0,len(L_T),step_size):
	
			# Add up labels
			A_T = L_T[counter:counter+frame_size, number_of_inputs]
			S_T = sum(A_T)
	
			if S_T>step_size:
				pos_examples_counter = pos_examples_counter + 1
				S_T = 1
			else:
				neg_examples_counter = neg_examples_counter + 1
				S_T = -1
	
			R_T = L_T[counter:counter+frame_size, :number_of_inputs]
	
			M_T = mean(R_T,axis=0)
			V_T = var(R_T,axis=0)
			SK_T = skew(R_T,axis=0)
			K_T = kurtosis(R_T,axis=0)
			RMS_T = sqrt(mean(R_T**2,axis=0))
	
			H_T = hstack((M_T,V_T))
			H_T = hstack((H_T,SK_T))
			H_T = hstack((H_T,K_T))
			H_T = hstack((H_T,RMS_T))
	
# ----------------------------- Label -------------------------------------
	
			# Add label
			H_T = hstack((H_T,S_T))
			if counter==0:
				F_T = H_T
			else:
				F_T = vstack((F_T,H_T))
	
		print ""
		print "Positive Examples: " + str(pos_examples_counter)
		print "Negative Examples: " + str(neg_examples_counter)
		print ""
	
		# Get features and labels
		X_T = F_T[:,:number_of_inputs*5]
		Y_T = F_T[:,number_of_inputs*5]
	
		print ""
		print "Shape of X_T: " +str(X_T.shape)
	
		print ""
		print "Shape of Y_T: " + str(Y_T.shape)
	
		X.append(X_T)
		Y.append(Y_T)
		
# --------------- Ground Truth - Get times for all activities --------------

		# Load annotated events into lists
		path = os.path.join(data_dir, 'participants', str(participant_counter), 'datafiles', 'annotations-sorted.csv')
		activities_time_i = []
		activities_eatingflag_i = []		
		with open(path, 'rb') as csvinputfile:
			csvreader = csv.reader(csvinputfile, delimiter=',', quotechar='|')

			print ""
			for row in csvreader:
				activities_time_i.append(int(sampling_rate * float(row[1]) / step_size))
				activities_eatingflag_i.append(2 * (int(row[2]) in eating_labels) - 1)
				print "GT Activity Time/Label: " + str(row[1]) + " " + str(row[2])
		activities_time.append(activities_time_i)
		activities_eatingflag.append(activities_eatingflag_i)
		
	dataset = {'data': {'X': X, 'Y': Y, 'sessions' : {'start' : activities_time, 'labels' : activities_eatingflag}}, \
		     'parameters' : {'frame_size': frame_size, 'step_size': step_size, 'units': units}}
	
	#save data to pickle file, if desired
	if save_pickle_path != 'none':
		if not save_pickle_path.endswith('.pickle'):
			load_pickle_path += '.pickle'	
		try:
			with open(save_pickle_path, 'wb') as handle:
				pickle.dump(dataset, handle)
		except:
			print("Failed to save data to " + str(load_pickle_path))
			pass
	return dataset

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--dir", dest="data_dir", default='../data/eating_detection_inertial_ubicomp2015/', \
			help="Directory where the dataset is stored.")	
	parser.add_argument("--frame-size", dest="frame_size", default=144, \
			help="The size of the sliding window over which instance feature-label pairs are defined.")
	parser.add_argument("--step-size", dest="step_size", default=72, \
			help="The step size of the sliding window.")
	parser.add_argument("--units", dest="units", default='u', \
			help="The units in which the frame size and step size are defined ('s' for seconds, 'u' for samples).")	
	parser.add_argument("--load", dest="load_pickle_path", default='none', \
			help="Path from which to load the pickle file. This will significantly speed up loading the data. " + \
			"If 'none' (default), the data will be reloaded from the specified directory.")	
	parser.add_argument("--save", dest="save_pickle_path", default='none', \
			help="Path where the data will be saved. If 'none' (default), the data will not be saved.")	
			
	args = parser.parse_args()

	dataset = load_data(**vars(args))
	
	print dataset