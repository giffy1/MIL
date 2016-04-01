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

matplotlib.use('Agg') #ensures plot can be viewed on server

from matplotlib import pyplot

participants = range(20)
K = range(20)

def main(working_dir, save_path):
	
	res_dir = working_dir + '/res'

	fscores = []
	for k in K:
		avg_fscore = 0
		count = 0
		for p in participants:
			res_path = os.path.join(res_dir, 'lopo_p%d_k%d.pickle' % (p, k))
			if os.path.isfile(res_path):
				with open(res_path, 'rb') as f:
					r = pickle.load(f)
				if not np.isnan(r['Results']['F1 Score']['Test']):
					avg_fscore += r['Results']['F1 Score']['Test']
					count += 1
		fscores.append(avg_fscore / count)
		
	pyplot.plot(K, fscores)
	pyplot.title("sbMIL performance varying number of instances from the held-out participant")
	pyplot.xlabel("number of instances")
	pyplot.ylabel("F1 Score")
	pyplot.savefig(save_path + 'test_instances.png')
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='/home/snoran/work/')	
	parser.add_argument("--save", dest="save_path", default='/home/snoran/work/res/plot', \
			help="Path of the plot file.")	
			
	args = parser.parse_args()

	main(**vars(args))