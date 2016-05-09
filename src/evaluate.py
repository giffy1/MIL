# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:21:20 2016

@author: snoran
"""

from __future__ import division

import os
import sys
from argparse import ArgumentParser
from bagging import main as bag_data
from lopo import main as lopo
import json

sys.path.insert(0, '../tests')
from qsub import qsub

#participants = range(1)
#N = 1000 # N : number of instances per participant put into bags
##M = [0, 12, 25, 50] # M : number of single-instance bags per participant
#bag_sizes = [1, 10, 20, 50, 100, 200]
#
#M=[250,500] #[0,125,250,500]
#
#local = True

#M = [0, 75, 150, 225, 300]
#bag_sizes = [1, 5, 10, 20, 40]

#for m=12, finish participants 11-19 with bag size 10
#for m=12 finish all participants with bag size 20
#for m=25, finish participants 15-19 with bag size 5 COMPLETE
#for m=25, finish all participants with bag sizes 10 and 20 COMPLETE
#do m=50

def main(working_dir, data_dir, n_jobs, n_trials, n_iter, bag_sizes, M, N, participants, local):

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
		
	M = json.loads(M)
	bag_sizes = json.loads(bag_sizes)
	try:
		participants = range(int(participants))
	except:
		participants = json.loads(participants)
	
	for p in participants:
		for m in M:
			for b in bag_sizes:
				for i in range(8,8+n_trials):
					file_str = '_p' + str(p) + '_m' + str(m) + '_b' + str(b) + '_i' + str(i)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					
					data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
					log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
					
					if local:
						bag_data(data_dir, data_file, b, p, m, N, i)
						lopo(data_file, 'sbMIL("verbose":0)', 'randomized', n_iter, n_jobs, 0, save_path, '')
					else:
#						submit_this_job = 'python bagging.py -d=%s -s=%s -p=%d -b=%d -m=%d -n=%d -i=%d' %(data_dir, data_file, p, b, m, N, i)
#						print submit_this_job + '\n'
#						bagging_job_id = 'bag' + file_str
#						qsub(submit_this_job, bagging_job_id, log_file, err_file, n_cores=n_jobs)	
						bag_data(data_dir, data_file, b, p, m, N, i)
						
						submit_this_job = 'python lopo.py -d=%s --n-jobs=%d --save=%s --n-iter=%d' %(data_file, n_jobs, save_path, n_iter)
						print submit_this_job + '\n'
						job_id = 'lopo3' + file_str
						qsub(submit_this_job, job_id, log_file, err_file, n_cores=n_jobs) #, depend=bagging_job_id)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--data-dir", dest="data_dir", \
		default='../data/smoking-data/', type=str, help="")
	parser.add_argument("-w", "--cwd", dest="working_dir", \
		default='eval_risq_m_b_final', type=str, help="")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, help="")
	parser.add_argument("--n-trials", dest="n_trials", default=12, type=int, help="")
	parser.add_argument("--n-iter", dest="n_iter", default=20, type=int, help="")
	parser.add_argument("-B", "--bag-sizes", dest="bag_sizes", default="[1,10,25,50,100,250]", type=str, help="")
	parser.add_argument("-M", "--n-single-instances", dest="M", default="[10,50,100,250,500]", type=str, help="")
	parser.add_argument("-N", "--n-bags", dest="N", default=1000, type=int, help="")
	parser.add_argument("-p", "--participants", dest="participants", default="[3]", type=str, help="")
	parser.add_argument("-l", "--local", dest="local", default=0, type=int, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))
