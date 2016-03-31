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

from matplotlib import pyplot

participants = range(19)
# number of positive sessions for each participant:
N = [8, 3, 15, 17, 4, 7, 4, 3, 9, 26, 41, 7, 5, 1, 3, 8, 3, 3, 3]

def main(working_dir, save_path):
	
	res_dir = working_dir + '/res'
	
	fscores = []
	for n in range(1,max(N)):
		for p in participants:
			avg_fscore = 0
			count = 0
			res_path = os.path.join(res_dir, 'lopo_p%d_n%d.pickle' % (p, n))
			if os.path.isfile(res_path):
				with open(res_path, 'rb') as f:
					r = pickle.load(f)
				avg_fscore += r['Results']['F1 Score']['Test']
				print("Avg F-score: %0.2f%%" %(100 * avg_fscore))
				count += 1
		if count == 0:
			fscores.append(0)
		else:
			fscores.append(avg_fscore / count)
		
	pyplot.figure()
	pyplot.plot(range(1,max(N)), fscores)
	pyplot.title("sbMIL performance varying number of positive sessions")
	pyplot.xlabel("number of bags N")
	pyplot.ylabel("F1 Score")
	pyplot.savefig(save_path + 'sessions.png')
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='/home/snoran/work/')	
	parser.add_argument("--save", dest="save_path", default='/home/snoran/work/res/plot', \
			help="Path of the plot file.")	
			
	args = parser.parse_args()

	main(**vars(args))