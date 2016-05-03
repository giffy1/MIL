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

sys.path.insert(0, '../tests')
from qsub import qsub

participants = range(1)
#M = [0, 12, 25, 50] # M : number of single-instance bags per participant
bag_sizes = [10]# [1,10,20,50,100] #[1, 10, 20, 50, 100]

#N = {1 : [0,25,50,75,100,125,150,175,200], 10 : range(0,101,10), 20: range(0,51,5), 50: range(0,20,4), 100: range(0,10,2)} # N : number of instances per participant put into bags

N = {10 : [70, 80, 90, 100]}
I = {70 : [4], 80: [1], 90: [0,1,2,4], 100 : [0]}

M=125

local = False

#M = [0, 75, 150, 225, 300]
#bag_sizes = [1, 5, 10, 20, 40]

#for b=10, n=20 finish participants 6-19
#complete all of b=20
#complete b=50,100 (currently on server)

def main(working_dir, data_dir, n_jobs, n_trials, n_iter):

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
	
	for b in bag_sizes:
		for n in N[b]:
			for p in participants:
				for i in I[n]: #range(n_trials):
					file_str = '_p' + str(p) + '_b' + str(b) + '_n' + str(n) 
					if n_trials > 1:
						file_str += '_i' + str(i)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					
					data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
					log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
					
					if local:
						bag_data(data_dir, data_file, b, p, M, n, i, shuffle_bags=True)
						lopo(data_file, 'sbMIL("verbose":0)', 'randomized', n_iter, n_jobs, 0, save_path, '')
					else:
#							submit_this_job = 'python bagging.py -d=%s -s=%s -p=%d -b=%d -m=%d -n=%d -i=%d' %(data_dir, data_file, p, b, M, n, i)
#							print submit_this_job + '\n'
#							bagging_job_id = 'bag' + file_str
#							qsub(submit_this_job, bagging_job_id, log_file, err_file, n_cores=n_jobs)
      
						bag_data(data_dir, data_file, b, p, M, n, i, shuffle_bags=True)
						
						submit_this_job = 'python lopo.py -d=%s --n-jobs=%d --save=%s --n-iter=%d' %(data_file, n_jobs, save_path, n_iter)
						print submit_this_job + '\n'
						job_id = 'lopo' + file_str
						qsub(submit_this_job, job_id, log_file, err_file, n_cores=n_jobs) #, depend=bagging_job_id)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--data-dir", dest="data_dir", \
		default='../data/eating_detection_inertial_ubicomp2015/', type=str, help="")
	parser.add_argument("-w", "--cwd", dest="working_dir", \
		default='eval5', type=str, help="")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, help="")
	parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, help="")
	parser.add_argument("--n-iter", dest="n_iter", default=8, type=int, help="")	
	
	args = parser.parse_args()
	
	main(**vars(args))