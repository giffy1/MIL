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

matplotlib.use('Agg') #ensures plot can be viewed on server

participants = range(19)

def main(working_dir):
	
	res_dir = working_dir + '/res'
	
	avg_fscore = 0
	count = 0
	for p in participants:
		res_path = os.path.join(res_dir, 'lopo_p%d.pickle' %p)
		if os.path.isfile(res_path):
			with open(res_path, 'rb') as f:
				r = pickle.load(f)
			avg_fscore += r['Results']['F1 Score']['Test']
			count += 1
	print("Average F1 Score : %0.2f%% " %(100*avg_fscore / count))
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='.')	
			
	args = parser.parse_args()

	main(**vars(args))