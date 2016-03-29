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

participants = range(20)
bag_sizes = [100,200,300]
N = {}
N[30] = range(0,81,5)
N[60] = range(0,41,4)
N[90] = range(0,37,3)
N[120] = range(0,25,3)
N[150] = range(0,19,2)
N[180] = range(0,15,2)
N[210] = range(0,13,2)
N[240] = range(0,9,1)
N[270] = range(0,7,1)
N[300] = range(0,7,1)

def main(working_dir, save_path):
	
	res_dir = working_dir + '/res'
	
	fscores = []
	for b in bag_sizes:
		fscores = []
		for n in N[b]:
			avg_fscore = 0
			count = 0
			for p in participants:
				res_path = os.path.join(res_dir, 'lopo_p%d_n%d_b%d.pickle' % (p, n, b))
				if os.path.isfile(res_path):
					with open(res_path, 'rb') as f:
						r = pickle.load(f)
					avg_fscore += r['Results']['F1 Score']['Test']
					count += 1
			if count == 0:
				fscores.append(0)
			else:
				fscores.append(avg_fscore / count)
		
		pyplot.figure()
		pyplot.plot(N[b], fscores)
		pyplot.title("sbMIL performance varying number of bags; bag-size = " + str(b))
		pyplot.xlabel("number of bags N")
		pyplot.ylabel("F1 Score")
		pyplot.savefig(save_path + '_b' + str(b) + '.png')
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='/home/snoran/work/')	
	parser.add_argument("--save", dest="save_path", default='/home/snoran/work/res/plot', \
			help="Path of the plot file.")	
			
	args = parser.parse_args()

	main(**vars(args))