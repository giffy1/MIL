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

participants = range(19)
K = range(50)

def main(working_dir, save_path):
	
	res_dir = working_dir + '/res'

	x = []
	fscores = []
	for k in K:
		avg_fscore = 0
		count = 0
		for p in participants:
			res_path = os.path.join(res_dir, 'lopo_p%d_k%d.pickle' % (p, k))
			if os.path.isfile(res_path):
				with open(res_path, 'rb') as f:
					r = pickle.load(f)
				print r['Results']['F1 Score']['Test']
				if not np.isnan(r['Results']['F1 Score']['Test']):
					avg_fscore += r['Results']['F1 Score']['Test']
					count += 1
		if count > 0:
			x.append(k)
			fscores.append(avg_fscore / count)
		
	pyplot.plot(x, fscores)
	pyplot.title("sbMIL performance varying number of sessions from the held-out subject")
	pyplot.xlabel("number of sessions")
	pyplot.ylabel("F1 Score")
	pyplot.savefig(save_path)
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='/home/snoran/work/')	
	parser.add_argument("--save", dest="save_path", default='/home/snoran/work/res/plot.png', \
			help="Path of the plot file.")	
			
	args = parser.parse_args()

	main(**vars(args))