# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 08:55:32 2016

@author: snoran
"""

import argparse
import os
import sys
from qsub import qsub
import itertools
		
class BaseEvaluator():
	
	def __init__(self, variables):
		self.variables = variables
		
		# parse remaining arguments from command-line input

		parser = argparse.ArgumentParser()
		parser.add_argument('-d', '--dir', dest='dir', default='.')
		parser.add_argument('--src', dest='src', default='../src')
		
		parser.add_argument("--data-dir", dest="data_dir", default='../../data/eating_detection_inertial_ubicomp2015', type=str, \
				help="Directory where the dataset is stored")
		
		parser.add_argument("--bag-size", dest="bag_size", default=-1, type=int, \
				help="If clf is an MIL classifier, bag-size specifies the size of each training bag")
		parser.add_argument("--held-out-bag-size", dest="held_out_bag_size", default=-1, type=int, \
				help=".")
		parser.add_argument("--test-bag-size", dest="test_bag_size", default=1, type=int, \
				help=".")	
	
		parser.add_argument("--N", dest="N", default=20, type=int, \
				help="Number of instances used for training in each LOPO iteration")
		parser.add_argument("--M", dest="M", default=100, type=int, \
				help="Number of single-instance bags used for training in each LOPO iteration")
		parser.add_argument("--K", dest="K", default=0, type=int, \
				help="Number of single-instance bags in the training data from the held-out participant.")
		
		parser.add_argument("--clf", dest="clf", default='sbMIL', type=str, \
				help="Classifier ('RF', 'SVM', 'LinearSVC', 'SIL', 'LinearSIL', 'MIForest', 'sMIL', 'sbMIL', 'misvm')")
		parser.add_argument("--eta", dest="eta", default=0.5, type=float, \
				help="Balancing parameter for sbMIL, between 0 and 1 inclusively")	
		parser.add_argument("--kernel", dest="kernel", default='linear', type=str, \
				help="Kernel type, i.e. 'linear', 'rbf', 'linear_av', etc.")
				
		parser.add_argument("--cv-method", dest="cv_method", default='randomized', type=str, \
				help="Determines how hyperparameters are learned ('grid' or 'randomized')")
		parser.add_argument("--cv", dest="cv", default=3, type=int, \
				help="Determines split for cross-validation (see GridSearchCV.cv)")
		parser.add_argument("--n-iter", dest="n_iter", default=10, type=int, \
				help="The number of iterations in randomized cross-validation (see RandomizedSearchCV.cv)")
		parser.add_argument("--n-trials", dest="n_trials", default=5, type=int, \
				help="Number of trials over which to average the performance metrics")
		parser.add_argument("--n-jobs", dest="n_jobs", default=1, type=int, \
				help="Number of threads used (default = 1). Use -1 for maximal parallelization")
		parser.add_argument("--n-jobs-per-iter", dest="n_jobs_per_iter", default=1, type=int, \
				help="Number of machines used for each run of GridSearchCV")
				
		parser.add_argument("--verbose", dest="verbose", default=1, type=int, \
				help="Indicates how much information should be reported (0=None, 1=Some, 2=Quite a bit)")	
		parser.add_argument("--desc", dest="desc", default='', type=str, \
				help="Description of the evaluation / parameter selection")
		
		self.args = parser.parse_args()
		self._mkdirs(self.args.dir)
		
	def _mkdirs(self, working_dir):
		"""
		Creates the working directory and its subdirectories if necessary. 
		These include err/, log/, params/ and res/ folders, which store 
		errors, logs, input parameters and results respecitively.
		
		@param working_dir : The working directory for this evaluation.
		
		Returns a list of the sub-directories.
		"""
		if not os.path.isdir(working_dir):
			os.mkdir(working_dir, 0755)
	
		self.log_dir = working_dir + '/log'
		if not os.path.isdir(self.log_dir):
			os.mkdir(self.log_dir, 0755)
		
		self.err_dir = working_dir + '/err'
		if not os.path.isdir(self.err_dir):
			os.mkdir(self.err_dir, 0755)
			
		self.res_dir = working_dir + '/res'
		if not os.path.isdir(self.res_dir):
			os.mkdir(self.res_dir, 0755)
			
		self.params_dir = working_dir + '/params'
		if not os.path.isdir(self.params_dir):
			os.mkdir(self.params_dir, 0755)
	
	def _get_number_of_participants(self):
		"""
		Returns the number of participants in the dataset found in the specified 
		data directory.
		"""
		sys.path.insert(0, self.args.data_dir)
		print os.getcwd()
		print sys.path
		from load_data import load_data
		dataset = load_data(self.args.data_dir)
		return len(dataset['data']['Y'])
		
	def _save_params(self):
		with open(os.path.join(self.params_dir, 'params.txt'), 'wb') as f:
			f.write(self.arg_str)
			
	def _tuple_to_str(self, keys, vals):
		string = ''
		for i, v in enumerate(vals):
			string += '_' + keys[i][0].lower() + str(v)
		return string
			
	def evaluate(self):
		self._get_base_arg_str()
		n_participants = self._get_number_of_participants()
		n_jobs_per_iter = self.args.n_jobs_per_iter
		if self.args.cv_method == 'grid':
			n_jobs_per_iter = 1
		n_iter = int(self.args.n_iter / n_jobs_per_iter)
		
		for p in range(n_participants):
			for i in range(1,n_jobs_per_iter+1):
				for vals in itertools.product(*self.variables.values()):
					var_arg_str = ''
					for j, var in enumerate(self.variables.keys()):
						var_arg_str += ' --' + var.replace('_', '-') + '="' + str(vals[j]) + '"'
					
					if n_jobs_per_iter > 1:
						file_str = '_p%d_i%d' %(p,i) + self._tuple_to_str(self.variables.keys(), vals)
					else:
						file_str = '_p%d' %p + self._tuple_to_str(self.variables.keys(), vals)
					
					save_path = os.path.join(self.res_dir, 'lopo' + file_str + '.pickle')
					submit_this_job = 'python %s/w_lopo.py --save=%s --test-participant=%d --cv=%d' % (self.args.src, save_path, p, n_iter) + self.arg_str + var_arg_str
					print submit_this_job + '\n'
					job_id = 'lopo' + file_str
					log_file = os.path.join(self.log_dir, 'log' + file_str + '.txt')
					err_file = os.path.join(self.err_dir, 'err' + file_str + '.txt')
					#qsub(submit_this_job, job_id, log_file, err_file, n_cores=self.args.n_jobs)
	
	def _get_base_arg_str(self):
		"""
		Returns the base argument string for evalating the model on the given dataset. 
		All parameters are defined here
		"""
		
		self.arg_str = ''
		for arg in self.args._get_kwargs():
			if arg[0] not in {'src', 'dir', 'cv', 'n_jobs_per_iter'} and arg[0] not in self.variables:
				self.arg_str += ' --' + arg[0].replace('_', '-') + '="' + str(arg[1]) + '"'
		self._save_params()