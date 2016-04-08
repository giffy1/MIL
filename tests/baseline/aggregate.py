# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 23:53:15 2016

@author: snoran
"""

from __future__ import division

import os
import pickle
import matplotlib
from argparse import ArgumentParser
import numpy as np
import sys

sys.path.insert(0, '../../src/')
from util import accuracy_precision_recall_fscore, pprint_header

matplotlib.use('Agg') #ensures plot can be viewed on server

participants = range(19)

def main(working_dir, verbose):
	
	res_dir = working_dir + '/res'
	
	total_conf = np.zeros((2,2))
	for p in participants:
		res_path = os.path.join(res_dir, 'lopo_p%d.pickle' %p)
		if os.path.isfile(res_path):
			with open(res_path, 'rb') as f:
				r = pickle.load(f)
			conf = r['Results']['Confusion Matrix']['Test']
			total_conf += conf
			
			if verbose:
				pprint_header("Participant: %d" %p)				
				
				fscore = r['Results']['F1 Score']['Test']
				precision = r['Results']['Precision']['Test']
				recall = r['Results']['Recall']['Test']
				print("Confusion Matrix:")
				print(conf)
				print("Precision: %0.2f%%" %(100*precision))
				print("Recall: %0.2f%%" %(100*recall))
				print("F1 Score: %0.2f%%" %(100*fscore))
		
	pprint_header("Aggregate Results:")			
	print("Total Confusion Matrix ")
	print(total_conf)
	_, prf = accuracy_precision_recall_fscore(total_conf)
	print("Average F1 Score : %0.2f%% " %(100*prf[1][2]))
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='.')
	parser.add_argument('-v', '--verbose', dest='verbose', default=1)
			
	args = parser.parse_args()

	main(**vars(args))