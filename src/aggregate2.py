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
	for f in files:
		_p_index = f.index("_p")
		_b_index = f.index("_b")
		_m_index = f.index("_n")
		_i_index = f.index("_i")
		p = int(f[_p_index+2:_b_index])
		b = int(f[_b_index+2:_m_index])
		n = int(f[_m_index+2:_i_index])
		
		if p in participants:
			with open(os.path.join(res_dir,f), 'rb') as fd:
				r = pickle.load(fd)
			conf = r["Results"]["Confusion Matrix"]["Test"]
			total_conf = confusion_matrix.get((b,n), np.asarray([[0,0],[0,0]]))
			confusion_matrix[(b,n)] = total_conf + conf
	for k,conf in confusion_matrix.iteritems():
		_, _, fscore = accuracy_precision_recall_fscore(conf)[1][1]
		fscores[k] = fscore
	keys = np.asarray(sorted(fscores.keys()))
	B = set(keys[:,0])
	B = [-1]
	for b in B:
		y=zip(*[(k[1],fscores[k]) for k in sorted(fscores.keys()) if k[0]==b])
		plt.plot(y[0][1:], y[1][1:], label="bag size " + str(b) if b>0 else "sessions")
	
	plt.xlabel("Number of bags")
	plt.ylabel("F1 Score")
	plt.title("Performance varying bag size and number of bags")
	plt.legend(loc=4)
	plt.show()
	
if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-w", "--cwd", dest="working_dir", default='eval_lab20_nbags', type=str, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))