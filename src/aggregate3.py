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

participants = range(20)

def main(working_dir):	
	res_dir = working_dir + '/res'

	files = [f for f in os.listdir(res_dir) if os.path.isfile(os.path.join(res_dir,f)) and f.startswith("lopo_") and f.endswith(".pickle")]
	confusion_matrix = {}
	fscores = {}
	stds = {}
	for f in files:
		_p_index = f.index("_p")
		_h_index = f.index("_h")
		_k_index = f.index("_k")
		_i_index = f.index("_i")
		_dot_index = f.index(".")
		p = int(f[_p_index+2:_h_index])
		h = int(f[_h_index+2:_k_index])
		k = int(f[_k_index+2:_i_index])
		i = int(f[_i_index+2:_dot_index])
		print p,h,k,i
		
		if p in participants:
			with open(os.path.join(res_dir,f), 'rb') as fd:
				r = pickle.load(fd)
			conf = r["Results"]["Confusion Matrix"]["Test"]
			total_conf = confusion_matrix.get((h,k), np.asarray([[0,0],[0,0]]))
			confusion_matrix[(h,k)] = total_conf + conf
			_, _, fscore = accuracy_precision_recall_fscore(conf)[1][1]
			if np.isnan(fscore):
				fscore = 0.0
			all_fscores = fscores.get((h,k), [])
			all_fscores.append(fscore)
			fscores[(h,k)]=all_fscores
	for k,conf in confusion_matrix.iteritems():
		stds[k] = np.std(fscores[k])
		_, _, fscore = accuracy_precision_recall_fscore(conf)[1][1]
#		if k[1]==0:
#			fscores[k] = 0.201 #baseline achieved when k=0
#		else:
		fscores[k] = fscore
		print k, fscore
	keys = np.asarray(sorted(fscores.keys()))
	H = set(keys[:,0])
	for h in H:
		y=zip(*[(k[1],fscores[k]) for k in sorted(fscores.keys()) if k[0]==h])
		plt.plot(y[0], y[1], label="bag size " + str(h) if h>0 else "sessions")
	
	plt.xlabel("Number of bags")
	plt.ylabel("F1 Score")
	plt.title("Performance varying bag size and number of bags")
	plt.legend(loc=4)
	plt.show()
	
if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-w", "--cwd", dest="working_dir", default='eval_lab20_nbags_heldout2', type=str, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))