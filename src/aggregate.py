# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:02:56 2016

@author: snoran
"""

from __future__ import division

import os
from argparse import ArgumentParser
import pickle
import numpy as np
from matplotlib import pyplot as plt
from util import accuracy_precision_recall_fscore

def main(working_dir):
	
	if not os.path.isdir(working_dir):
		os.mkdir(working_dir, 0755)
	
	log_dir = working_dir + '/log'
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir, 0755)
	
	err_dir = working_dir + '/err'
	if not os.path.isdir(err_dir):
		os.mkdir(err_dir, 0755)
		
	res_dir = working_dir + '/res'
	if not os.path.isdir(res_dir):
		os.mkdir(res_dir, 0755)
		
	files = [f for f in os.listdir(res_dir) if os.path.isfile(os.path.join(res_dir,f)) and f.startswith("lopo_") and f.endswith(".pickle")]
	
	confusion_matrix = {}
	fscores = {}
	for f in files:
		#_p_index = f.index("_p")
		_b_index = f.index("_b")
		_m_index = f.index("_m")
		_i_index = f.index("_i")
		#p = int(f[_p_index+2:_b_index])
		m = int(f[_m_index+2:_b_index])
		b = int(f[_b_index+2:_i_index])
		
		with open(os.path.join(res_dir,f), 'rb') as fd:
			r = pickle.load(fd)
		conf = r["Results"]["Confusion Matrix"]["Test"]
		total_conf = confusion_matrix.get((m,b), np.asarray([[0,0],[0,0]]))
		confusion_matrix[(m,b)] = total_conf + conf
	for k,conf in confusion_matrix.iteritems():
		_, _, fscore = accuracy_precision_recall_fscore(conf)[1][1]
		fscores[k] = fscore
		print k, fscore
	keys = np.asarray(sorted(fscores.keys()))
	M = set(keys[:,0])
	print M
	for m in M:
		y=zip(*[(k[1],fscores[k]) for k in sorted(fscores.keys()) if k[0]==m])
		plt.plot(y[0], y[1], label="M=" + str(m))
	
	plt.xlabel("Bag Size")
	plt.ylabel("F1 Score")
	plt.title("Performance varying number of single instances and bag size")
	plt.legend(loc=4)
	plt.show()
	
if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-w", "--cwd", dest="working_dir", \
		default='eval4', type=str, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))