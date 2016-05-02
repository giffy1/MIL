# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:21:20 2016

@author: snoran
"""

from __future__ import division

import os
import sys
from argparse import ArgumentParser
import pickle
import numpy as np
from matplotlib import pyplot as plt
from util import pprint_header, accuracy_precision_recall_fscore
from bagging import main as bag_data
from lopo import main as lopo

sys.path.insert(0, '../tests')
from qsub import qsub

participants = range(18)
N = 1000 # N : number of instances per participant put into bags
#M = [0, 12, 25, 50] # M : number of single-instance bags per participant
bag_sizes = [1, 10, 20, 50, 100, 200]

M=[0,125,250,500]

local = True

#M = [0, 75, 150, 225, 300]
#bag_sizes = [1, 5, 10, 20, 40]

#for m=12, finish participants 11-19 with bag size 10
#for m=12 finish all participants with bag size 20
#for m=25, finish participants 15-19 with bag size 5 COMPLETE
#for m=25, finish all participants with bag sizes 10 and 20 COMPLETE
#do m=50

def main(aggregate, working_dir, data_dir, n_jobs, n_trials, n_iter):

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
  
      	plot_dir = res_dir + '/plots'
	if not os.path.isdir(plot_dir):
		os.mkdir(plot_dir, 0755)
	
	for p in participants:
		handles = []
		if aggregate:
			plt.figure()
		for m in M:
			fscores = []
			for b in bag_sizes:
				if aggregate:
					pprint_header("Aggregating Results for participant %d, M = %d, bag size = %d" %(p, m, b))
				total_conf = np.asarray([[0,0],[0,0]])
				
				for i in range(n_trials):
					file_str = '_p' + str(p) + '_m' + str(m) + '_b' + str(b) 
					if n_trials > 1:
						file_str += '_i' + str(i)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					
					if not aggregate:					
						data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
						log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
						err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
						
						if local:
							bag_data(data_dir, data_file, b, p, m, N, i)
							lopo(data_file, 'sbMIL("verbose":0)', 'randomized', n_iter, n_jobs, 0, save_path, '')
						else:
#							submit_this_job = 'python bagging.py -d=%s -s=%s -p=%d -b=%d -m=%d -n=%d -i=%d' %(data_dir, data_file, p, b, m, N, i)
#							print submit_this_job + '\n'
#							bagging_job_id = 'bag' + file_str
#							qsub(submit_this_job, bagging_job_id, log_file, err_file, n_cores=n_jobs)	
							bag_data(data_dir, data_file, b, p, m, N, i)
							
							submit_this_job = 'python lopo.py -d=%s --n-jobs=%d --save=%s --n-iter=%d' %(data_file, n_jobs, save_path, n_iter)
							print submit_this_job + '\n'
							job_id = 'lopo' + file_str
							qsub(submit_this_job, job_id, log_file, err_file, n_cores=n_jobs) #, depend=bagging_job_id)
					else:
						if os.path.isfile(save_path):
							with open(save_path, 'rb') as f:
								r = pickle.load(f)
							conf = r["Results"]["Confusion Matrix"]["Test"]
							total_conf += conf
				if aggregate:
					print(total_conf)
					precision, recall, fscore = accuracy_precision_recall_fscore(total_conf)[1][1]
					print("F1 score: %0.02f" %fscore)
					fscores.append(fscore)
			if aggregate:
				h, = plt.plot(bag_sizes, fscores, label="M=" + str(m))
				handles.append(h)
		if aggregate:
			plt.xlabel("Bag size")
			plt.ylabel("F1 Score")
			plt.title("Performance varying bag size and number of single instances")
			plt.legend(handles = handles)
			plt.savefig(os.path.join(plot_dir, "p%d_b_m.png" %p))

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--data-dir", dest="data_dir", \
		default='../data/smoking-data/', type=str, help="")
	parser.add_argument("-w", "--cwd", dest="working_dir", \
		default='eval11', type=str, help="")
	parser.add_argument("-a", "--aggregate", dest="aggregate", default=0, type=int, help="")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, help="")
	parser.add_argument("--n-trials", dest="n_trials", default=3, type=int, help="")
	parser.add_argument("--n-iter", dest="n_iter", default=10, type=int, help="")	
	
	args = parser.parse_args()
	
	main(**vars(args))