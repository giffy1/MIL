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

sys.path.insert(0, '../tests')
from qsub import qsub

participants = range(20)
data_dir = '../data/eating_detection_inertial_ubicomp2015/'
N = 100 # N : number of instances per participant put into bags
M = [0, 25, 50, 75, 100] # M : number of single-instance bags per participant
bag_sizes = [1,5,10,20]

working_dir = '.'

def main(aggregate):

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
	
	for m in M:
		for b in bag_sizes:
			for p in participants:
				file_str = '_p' + str(p) + '_b' + str(b) + '_m' + str(m)
				save_path = os.path.join(res_dir, 'lopo' + file_str + '.pickle')
				if aggregate:
					if os.path.isfile(save_path):
						with open(save_path, 'rb') as f:
							r = pickle.load(f)
							print(r["Results"]["F1 Score"]["Test"])
				else:
					data_file = os.path.join(res_dir, 'data' + file_str + '.pickle')
					bag_data(data_dir, data_file, b, p, m, N)
					
					submit_this_job = 'python lopo.py -d=%s --n-jobs=3 --save=%s' %(data_file, save_path)
					print submit_this_job + '\n'
					job_id = 'lopo' + file_str
					log_file = os.path.join(log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(err_dir, 'err' + file_str + '.txt')
					qsub(submit_this_job, job_id, log_file, err_file, n_cores=3)
			
if __name__ == "__main__":
	parser = ArgumentParser()
	
	parser.add_argument("-a", dest="aggregate", default=True, type=bool, help="")
	
	args = parser.parse_args()
	
	main(**vars(args))