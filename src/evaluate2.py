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
	
	N = json.loads(N)
	bag_sizes = json.loads(bag_sizes)
	try:
		participants = range(int(participants))
	except:
		participants = json.loads(participants)
	
	bag_sizes = [1,5,10,20,50,100,200]
	N = {1: range(0,101,10), 5: range(0,101,10), 10: range(0,51,5), 20: range(0,31,3), 50: range(11), 100: range(6), 200: range(3)}
	
	for b in bag_sizes:
		for n in N[b]:
			for p in participants:
				for i in range(25,25+n_trials):
					file_str = '_p' + str(p) + '_b' + str(b) + '_n' + str(n) + '_i' + str(i)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					
					data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
					log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
					
					if local:
						bag_data(data_dir, data_file, b, p, M, n, i, shuffle_bags=True, shuffle_si=False)
						lopo(data_file, 'sbMIL("verbose":0)', 'randomized', n_iter, n_jobs, 0, save_path, '')
					else:
						#don't submit jobs for bagging, the overhead is too large:
#						submit_this_job = 'python bagging.py -d=%s -s=%s -p=%d -b=%d -m=%d -n=%d -i=%d' %(data_dir, data_file, p, b, M, n, i)
#						print submit_this_job + '\n'
#						bagging_job_id = 'bag' + file_str
#						qsub(submit_this_job, bagging_job_id, log_file, err_file, n_cores=n_jobs)
      
						bag_data(data_dir, data_file, b, p, M, n, i, shuffle_bags=True, shuffle_si=False)
						
						submit_this_job = 'python lopo.py -d=%s --n-jobs=%d --save=%s --n-iter=%d' %(data_file, n_jobs, save_path, n_iter)
						print submit_this_job + '\n'
						job_id = 'lopo3' + file_str
						qsub(submit_this_job, job_id, log_file, err_file, n_cores=n_jobs) #, depend=bagging_job_id)

# finish p=0, b=50, n=10, trials 2,3,4
# then finish p=0, b=50, n=15, all trials (being done on server now)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d", "--data-dir", dest="data_dir", \
		default='../data/eating_detection_inertial_ubicomp2015/', type=str, help="") #eating_detection_inertial_ubicomp2015
	parser.add_argument("-w", "--cwd", dest="working_dir", \
		default='lopo_lab20_nbags_m125', type=str, help="")
	parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, help="")
	parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, help="")
	parser.add_argument("--n-iter", dest="n_iter", default=25, type=int, help="")	
	parser.add_argument("-B", "--bag-sizes", dest="bag_sizes", default="[-1,1,5,10,20,50,100,200]", type=str, help="")
	parser.add_argument("-M", "--n-single-instances", dest="M", default=125, type=int, help="")
	parser.add_argument("-N", "--n-bags", dest="N", default="[0,10,20,30,40,50]", type=str, help="")
	parser.add_argument("-p", "--participants", dest="participants", default="20", type=str, help="")
	parser.add_argument("-l", "--local", dest="local", default=0, type=int, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))