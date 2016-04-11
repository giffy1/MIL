# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:08:35 2016

@author snoran
"""

import numpy as np
from argparse import ArgumentParser
import sys

from matplotlib import pyplot as plt

parser = ArgumentParser()
	
parser.add_argument("--data-dir", dest="data_dir", default='../data/smoking-data', type=str, \
			  help="Directory where the dataset is stored")

args = parser.parse_args()

sys.path.insert(0, args.data_dir)
from load_data import load_data

def plot_histogram(lst, n_bins, title, xlabel, ylabel):
	plt.figure()
	hist, bins = np.histogram(lst, bins = n_bins)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

dataset = load_data(args.data_dir)
X = dataset['data']['X']
Y = dataset['data']['Y']
session_start = dataset['data']['sessions']['start']
session_labels = dataset['data']['sessions']['labels']

# %%---------------------------------------------------------------------------
#
#			    Plot Histogram of Dataset Size
#
# -----------------------------------------------------------------------------

lengths = [len(X[k]) for k in range(len(X))]
plot_histogram(lengths, 6, "Distribution of Dataset Size over Participants", \
		  "Number of Labeled Instances", "Number of Participants")

# %%---------------------------------------------------------------------------
#
#		   Plot Histogram of Fracion of Positive Instances
#
# -----------------------------------------------------------------------------

fraction_of_positive_instances = [np.mean(np.greater(Y[k], 0)) for k in range(len(Y))]
plot_histogram(fraction_of_positive_instances, 10, "Distribution of Fraction of Positive Instances over Participants", \
		  "Fraction of Positive Instances", "Number of Participants")

# %%---------------------------------------------------------------------------
#
#		   Plot Histogram of Fracion of Positive Sessions
#
# -----------------------------------------------------------------------------

fraction_of_positive_sessions = [np.mean(np.greater(session_labels[k], 0)) for k in range(len(Y))]
plot_histogram(fraction_of_positive_sessions, 5, "Distribution of Fraction of Positive Sessions over Participants", \
		  "Fraction of Positive Sessions", "Number of Participants")

# %%---------------------------------------------------------------------------
#
#		 Plot Histogram of Average Session Duration per Participant
#
# -----------------------------------------------------------------------------

session_durations = [np.hstack((session_start[k][1:],len(X[k])))-session_start[k] for k in range(len(X))]
avg_session_durations = [np.mean(session_durations[k]) for k in range(len(session_durations))]
plot_histogram(avg_session_durations, 5, "Distribution of Average Session Duration over Participants", \
		  "Average Session Duration (# instances)", "Number of Participants")
				
# %%---------------------------------------------------------------------------
#
#		 Plot Histogram of Average Session Duration per Participant
#
# -----------------------------------------------------------------------------

fraction_of_positive_instances_per_session = []
for p in range(len(X)):
	for i in range(len(session_start[p])):
		if session_labels[p][i] > 0:
			avg = np.mean(np.greater(Y[p][session_start[p][i]:session_start[p][i] + session_durations[p][i]], 0))
			fraction_of_positive_instances_per_session.append(avg) 
plot_histogram(fraction_of_positive_instances_per_session, 10, "Distribution of Fraction of Positive Instances in Positive Sessions", \
		  "Fraction of Positive Instances", "Number of Sessions")