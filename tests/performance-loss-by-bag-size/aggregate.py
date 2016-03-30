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
M = range(0,501,50)
bag_size = [1,60,120,180,240,300]

def main(working_dir, save_path):
	
	res_dir = working_dir + '/res'
	

	for m in M:
		fscores = []
		for b in bag_size:
			avg_fscore = 0
			count = 0
			for p in participants:
				res_path = os.path.join(res_dir, 'lopo_p%d_m%d_b%d.pickle' % (p, m, b))
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
		pyplot.plot(M, fscores)
		pyplot.title("sbMIL performance varying bag size: M = " + str(M))
		pyplot.xlabel("Bag Size")
		pyplot.ylabel("F1 Score")
		pyplot.savefig(save_path + '_M' + str(M) + '.png')
	

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-d', '--dir', dest='working_dir', default='.')	
	parser.add_argument("--save", dest="save_path", default='./res/plots/plot.png', \
			help="Path of the plot file.")	
			
	args = parser.parse_args()

	main(**vars(args))