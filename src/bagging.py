# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:50:49 2016

@author: snoran
"""

from util import shuffle
import sys
import pickle
from argparse import ArgumentParser

MIL = {'SIL', 'sMIL', 'sbMIL', 'misvm', 'MIForest'}

def single_instances_to_sessions(X, Y, session_labels, session_start):
	"""
	TODO: Doc String
	"""
	bags = []
	labels = []
	single_instance_labels = []
	for j in range(len(session_labels)):
		if j < len(session_labels)-1:
			end = session_start[j+1]
		else:
			end = len(X)
		bags.append(X[session_start[j]:end, :])
		single_instance_labels.append(Y[session_start[j]:end])
		labels.append(session_labels[j])
			
	return bags, labels, single_instance_labels
	
def main(data_dir, data_file, bag_size, active_participant_counter, M, N, seed=None):

#data_dir = '../data/eating_detection_inertial_ubicomp2015/'
#data_dir = '../data/smoking-data/'
#data_file = "data_p0.pickle"

	sys.path.insert(0, data_dir)
	from load_data import load_data
	
	dataset = load_data(data_dir)
	X = dataset['data']['X']
	Y = dataset['data']['Y']
	session_start = dataset['data']['sessions']['start']
	session_labels = dataset['data']['sessions']['labels']	
	
	participant_indices = range(len(X))
	n_si_participants = 5
	n_bag_participants = len(X) - n_si_participants - 1
	
	#indices for participants in training data; skip active participant counter:
	train_indices = participant_indices[:active_participant_counter] + participant_indices[active_participant_counter+1:]	
	
	si_participant_indices = train_indices[:n_si_participants]
	bag_participant_indices = train_indices[n_si_participants:n_si_participants+n_bag_participants+1]
		
	#single-instance training data:
	X_SI = []
	Y_SI = []
	for p in si_participant_indices:
		x = X[p]
		y = Y[p]
		x,y = shuffle(seed, x, y)
		X_SI.append(x)
		Y_SI.append(y)
	
	#bag-level training data:
	X_B = []
	Y_B = []
	for p in bag_participant_indices:
		if bag_size == -1:
			x, y, _ = single_instances_to_sessions(X[p], Y[p], session_labels[p], session_start[p])
		else:
			x = [X[p][k:k+bag_size, :] for k in xrange(0, min(len(X[p]), N), bag_size)]
			y = [max(Y[p][k:k+bag_size]) for k in xrange(0, min(len(Y[p]), N), bag_size)]
		#x,y = shuffle(x,y)	
		X_B.append(x)
		Y_B.append(y)
		
	#training data from the held-out participant:
	#TODO: ^
	
	#test data:
	X_test = X[active_participant_counter]
	Y_test = Y[active_participant_counter]
	#X_test, Y_test = shuffle(X_test, Y_test)

##convert to bags:
#if clf_name in MIL:
#	X_SI = [X_SI[k:k+1, :] for k in xrange(len(X_SI))]
#	Y_SI = [max(Y_SI[k:k+1]) for k in xrange(len(Y_SI))]
#	
#	if bag_size == -1:
#		X_B, Y_B, _ = single_instances_to_sessions(X, Y, session_labels, session_start, bag_participant_indices)
#	else:
#		X_B = [X_B[k:k+bag_size, :] for k in xrange(0, len(X_B), bag_size)]
#		Y_B = [max(Y_B[k:k+bag_size]) for k in xrange(0, len(Y_B), bag_size)]
#	
#	if held_out_bag_size == -1:
#		X_T, Y_T, Y_si = single_instances_to_sessions(X, Y, session_labels, session_start, [active_participant_counter])
#	else:
#		X_T = [X_test[k:k+bag_size, :] for k in xrange(0,len(X_test), held_out_bag_size)]
#		Y_si = [Y_test[k:k+bag_size] for k in xrange(0,len(Y_test), held_out_bag_size)]
#		Y_T = [max(y_t) for y_t in Y_si]
#						
#	X_T, Y_T, Y_si = shuffle(X_T, Y_T, Y_si)	
#
#	# convert remaining bags back to test instances
#	X_test = []
#	Y_test = []
#	for i, (x_t, y_si) in enumerate(zip(X_T, Y_si)[K:]):
#		for (x,y) in zip(x_t, y_si):
#			X_test.append(x)
#			Y_test.append(y)
#			
#	X_test = [np.asarray(X_test)[k:k+test_bag_size, :] for k in xrange(0, len(X_test), test_bag_size)]
#	Y_test = [max(Y_test[k:k+test_bag_size]) for k in xrange(0, len(Y_test), test_bag_size)]
#
#else: # standard supervised learning case
#	X_T = X_test[:K]
#	X_T = X_test[:K]
#	X_test = X_test[K:]
#	Y_test = Y_test[K:]
#	
#if N < 0:
#	N=len(X_B)
#	
#if M < 0:
#	M=len(X_SI)
#		
#X_SI, Y_SI = shuffle(X_SI, Y_SI)
#X_B, Y_B = shuffle(X_B, Y_B)
#X_test, Y_test = shuffle(X_test, Y_test)

	data = {}
	data['training'] = {'instance' : {'X' : X_SI, 'Y' : Y_SI, 'M': M}, 'bag' : {'X' : X_B, 'Y' : Y_B}}
	data['test'] = {'X' : X_test, 'Y' : Y_test}
	
	with open(data_file, 'wb') as f:
		pickle.dump(data, f)
		
	return data

# print number of bags per participant	
# [len(X_B[k]) for k in range(len(X_B))]

# print size of bags for a particular subject
# [len(X_B[p][k]) for k in range(len(X_B[p]))]
	
# print number of instances per participant	
# [len(X_SI[k]) for k in range(len(X_SI))]

if __name__ == "__main__":

	parser = ArgumentParser()
	
	parser.add_argument("-d", "--data-dir", dest="data_dir", default='../data/eating_detection_inertial_ubicomp2015/', type=str, \
			help="Directory where the dataset is stored.")
	parser.add_argument("-s", "--save", dest="data_file", default='data.pickle', type=str, \
			help="File where the bagged data will be stored.")
	parser.add_argument("-p", "--participant", dest="active_participant_counter", default=3, type=int, \
			help="Participant held out for evaluating the model.")	
	parser.add_argument("-b", "--bag-size", dest="bag_size", default=0, type=int, \
			help="Bag Size (-1 for sessions)")
	parser.add_argument("-m", "--M", dest="M", default=75, type=int, \
			help="")
	parser.add_argument("-n", "--N", dest="N", default=100, type=int, \
			help="")
	parser.add_argument("-i", "--seed", dest="seed", default=0, type=int, \
			help="")
	
	args = parser.parse_args()
	
	data = main(**vars(args))