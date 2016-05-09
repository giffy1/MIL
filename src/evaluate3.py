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

def main(working_dir, data_dir, n_jobs, n_trials, n_iter, bag_size, M, N, participants, local):
	
#	held_out_bag_sizes = [-1] #[-1,1,5,10,50,100]
#	K = {-1 : range(0,10)} #, 1 : range(0,101,20), 5 : range(0,41,4), 10 : range(0,21,2), 20: range(0,11,2), 50:range(5), 100:range(3)}
	
#	held_out_bag_sizes = [5,10,50,100]
#	K = {1 : range(0,201,20), 5 : range(0,41,4), 10 : range(0,21,2), 20: range(0,11,2), 50:range(5), 100:range(3)}

	held_out_bag_sizes = [-1,1,5,10,20,50,100]
	K = {-1 : range(5), 1 : range(0,201,40), 5 : range(0,41,8), 10 : range(0,21,4), 20: range(0,11,2), 50:range(5), 100:range(3)}
	K_max = 200

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
	
	try:
		participants = range(int(participants))
	except:
		participants = json.loads(participants)
	
	for p in participants:
		for h in held_out_bag_sizes:
			for k in K[h]:
				for i in range(18,25):
					file_str = '_p' + str(p) + '_h' + str(h) + '_k' + str(k) + '_i' + str(i)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					
					data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
					log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
										
					if local:
						bag_data(data_dir, data_file, bag_size, p, M, N, i, shuffle_bags=False, shuffle_si=False, K=k, K_max = K_max, held_out_b=h, shuffle_heldout=True)
						lopo(data_file, 'sbMIL("verbose":0)', 'randomized', n_iter, n_jobs, 0, save_path, '')
					else:
#						#don't submit jobs for bagging, the overhead is too large:
#						submit_this_job = 'python bagging.py -d=%s -s=%s -p=%d -b=%d -m=%d -n=%d -i=%d' %(data_dir, data_file, p, b, M, n, i)
#						print submit_this_job + '\n'
#						bagging_job_id = 'bag' + file_str
#						qsub(submit_this_job, bagging_job_id, log_file, err_file, n_cores=n_jobs)
      
						bag_data(data_dir, data_file, bag_size, p, M, N, i, shuffle_bags=False, shuffle_si=False, K=k, K_max = K_max, held_out_b=h, shuffle_heldout=True)
						
						submit_this_job = 'python lopo.py -d=%s --n-jobs=%d --save=%s --n-iter=%d' %(data_file, n_jobs, save_path, n_iter)
						print submit_this_job + '\n'
						job_id = 'lopo5' + file_str
						qsub(submit_this_job, job_id, log_file, err_file, n_cores=n_jobs) #, depend=bagging_job_id)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--data-dir", dest="data_dir", \
		default='../data/eating_detection_inertial_ubicomp2015/', type=str, help="")
	parser.add_argument("-w", "--cwd", dest="working_dir", \
		default='eval_lab20_nsessions_heldout1', type=str, help="")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, help="")
	parser.add_argument("--n-trials", dest="n_trials", default=18, type=int, help="")
	parser.add_argument("--n-iter", dest="n_iter", default=20, type=int, help="")	
	parser.add_argument("-b", "--bag-size", dest="bag_size", default=10, type=int, help="")
	parser.add_argument("-m", "--n-single-instances", dest="M", default=125, type=int, help="")
	parser.add_argument("-n", "--n-bags", dest="N", default=10, type=int, help="")
	parser.add_argument("-p", "--participants", dest="participants", default="[0]", type=str, help="")
	parser.add_argument("-l", "--local", dest="local", default=0, type=int, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))