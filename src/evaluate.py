# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:21:20 2016

@author: snoran
"""

from bagging import main as bag_data
import os
import sys
from argparse import ArgumentParser
import pickle
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, '../tests')
from qsub import qsub

participants = range(20)
data_dir = '../data/eating_detection_inertial_ubicomp2015/'
N = 100 # N : number of instances per participant put into bags
M = [0, 25, 50, 75, 100] # M : number of single-instance bags per participant
bag_sizes = [1,5,10,20]

working_dir = '.'

def main(aggregate, n_jobs):

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
	
	print aggregate
	print aggregate==True
	if aggregate:	
		handles = []
		for m in M:
			fscores = []
			for b in bag_sizes:
				avg_fscore = 0
				participant_count = 0
				for p in participants:
					file_str = '_p' + str(p) + '_b' + str(b) + '_m' + str(m)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					if os.path.isfile(save_path):
						with open(save_path, 'rb') as f:
							r = pickle.load(f)
							fscore = r["Results"]["F1 Score"]["Test"]
							if not np.isnan(fscore):
								avg_fscore += fscore
								participant_count += 1
						
				fscores.append(avg_fscore / participant_count)
			h, = plt.plot(bag_sizes, fscores, label="M=" + str(m))
			handles.append(h)
		plt.xlabel("Bag size")
		plt.ylabel("F1 Score")
		plt.title("Performance varying bag size and number of single instances")
		plt.legend(handles = handles)
		plt.show()
	else:
		for m in M:
			fscores = []
			for b in bag_sizes:
				for p in participants:
					file_str = '_p' + str(p) + '_b' + str(b) + '_m' + str(m)
					save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
					data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
					bag_data(data_dir, data_file, b, p, m, N)
					
					submit_this_job = 'python lopo.py -d=%s --n-jobs=%d --save=%s --n-iter=%d' %(data_file, n_jobs, save_path, 25)
					print submit_this_job + '\n'
					job_id = 'lopo' + file_str
					log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
					qsub(submit_this_job, job_id, log_file, err_file, n_cores=n_jobs)
			
if __name__ == "__main__":
	parser = ArgumentParser()
	
	parser.add_argument("-a", dest="aggregate", default=1, type=int, help="")
	parser.add_argument("-n-jobs", dest="n_jobs", default=1, type=int, help="")	
	
	args = parser.parse_args()
	
	main(**vars(args))