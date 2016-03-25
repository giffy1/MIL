# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:15:52 2016

@author: snoran
"""

"""
Wrapper class for C++ MIForest implementation
"""
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
import subprocess
import scipy
import collections
import os

def update_config(filename, directory, prefix, n_estimators):
	with open(filename, "rb") as f:
		conf = f.read()
	
	path = os.path.join(directory, str(prefix) + ".data")
	conf = conf[:conf.index("data_file")] + 'data_file = "' + str(path) + '";\n\t' + conf[conf.index("sample_labels"):]  

	path = os.path.join(directory, str(prefix) + "-instance.labels")
	conf = conf[:conf.index("sample_labels")] + 'sample_labels = "' + str(path) + '";\n\t' + conf[conf.index("bag_sample_indices"):] 
	
	path = os.path.join(directory, str(prefix) + "-instance.index")
	conf = conf[:conf.index("bag_sample_indices")] + 'bag_sample_indices = "' + str(path) + '";\n\t' + conf[conf.index("bag_labels"):]
	
	path = os.path.join(directory, str(prefix) + "-bag.labels")
	conf = conf[:conf.index("bag_labels")] + 'bag_labels = "' + str(path) + '";\n\t' + conf[conf.index("train_bag_indices"):]
	
	path = os.path.join(directory, str(prefix) + "-train-bag.index")
	conf = conf[:conf.index("train_bag_indices")] + 'train_bag_indices = "' + str(path) + '";\n\t' + conf[conf.index("test_bag_indices"):]
	
	path = os.path.join(directory, str(prefix) + "-test-bag.index")
	conf = conf[:conf.index("test_bag_indices")] + 'test_bag_indices = "' + str(path) + '";\n\t' + conf[conf.index("do_sample_weighting"):]
	
	conf = conf[:conf.index("forest_size")] + 'forest_size = ' + str(n_estimators) + ';\n\t' + conf[conf.index("train_sampling"):]  

	with open(filename, "wb") as f:
		f.write(conf)

def write_data(dtype, values, n_classes, path):
	'''
	Writes a matrix to a data file in the format specified by the MIForest implementation
	'''
	with open(path, 'wb') as f:
		if scipy.sparse.issparse(values):
			sparsity="sparse"
		else:
			sparsity="dense"
		f.write("\n".join([dtype, ' '.join(str(x) for x in values.shape + (n_classes,)), sparsity]))
		
		for row in values:
			if isinstance(row, collections.Iterable):
				f.write("\n" + ' '.join([str(v) for v in row]))
			else:
				f.write("\n" + str(row))
		f.write("\n")

def write_mil_data(train_bags, train_bag_labels, test_bags, test_bag_labels, directory, prefix):
	bags = train_bags + test_bags
	bag_labels = 1*np.greater(np.hstack((train_bag_labels, test_bag_labels)),0).astype(int)
	
	train_instances = np.vstack(train_bags)
	train_instance_labels = np.vstack([(label>0) * np.ones((len(bag), 1)) for bag, label in zip(train_bags, train_bag_labels)]).astype(int)
	
	test_instances = np.vstack(test_bags)
	test_instance_labels = np.vstack([(label>0) * np.ones((len(bag), 1)) for bag, label in zip(test_bags, test_bag_labels)]).astype(int)
	
	instances = np.vstack((train_instances, test_instances))
	instance_labels = np.vstack((train_instance_labels, test_instance_labels)).astype(int)
	
	instance_index = np.vstack([index * np.ones((len(bag), 1)) for index, bag in enumerate(bags)]).astype(int)
			
	path = os.path.join(directory, str(prefix) + ".data")
	write_data("double", instances, 2, path)	

	path = os.path.join(directory, str(prefix) + "-instance.labels")
	write_data("int", instance_labels, 2, path)																							

	path = os.path.join(directory, str(prefix) + "-instance.index")
	write_data("int", instance_index, 2, path)

	path = os.path.join(directory, str(prefix) + "-bag.labels")
	write_data("int", np.reshape(bag_labels, (-1,1)), 2, path)
	
	path = os.path.join(directory, str(prefix) + "-train-bag.index")
	write_data("int", np.reshape(range(len(train_bags)), (-1, 1)), 2, path)
	
	path = os.path.join(directory, str(prefix) + "-test-bag.index")
	write_data("int", np.reshape(range(len(train_bags), len(bags)), (-1, 1)), 2, path)

class MIForest(ClassifierMixin, BaseEstimator):
	"""
	Multi-Instance Random Forest Implementation.
	"""

	def __init__(self, directory, prefix='_', n_estimators=50):
		self.directory = directory
		self.prefix = prefix
		
		self.train_bags = []
		self.train_bag_labels = []
		self.test_bags = []
		self.test_bag_labels = []
		
		self.n_estimators = n_estimators


	def fit(self, X, y):
		"""
		@param X : an n-by-m array-like object containing n examples with m features
		@param y : an array-like object of length n containing -1/+1 labels
		"""
		self.train_bags = X
		self.train_bag_labels = y
		return self
		
	def predict(self, X):
		self.test_bags = X
		self.test_bag_labels = -np.ones(len(X)) #Arbitrary labels to fit file data format
		
		data_dir = os.path.join(self.directory, "data")
		config_file = os.path.join(self.directory, "config.conf")
		miforest_exe = os.path.join(self.directory, "MIL-Forest")
		
		write_mil_data(self.train_bags, self.train_bag_labels, self.test_bags, self.test_bag_labels, data_dir, self.prefix)
		update_config(config_file, data_dir, self.prefix, self.n_estimators)
		
		args = (miforest_exe, config_file)
		#Or just:
		#args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
		popen = subprocess.Popen(args, stdout=subprocess.PIPE)
		ypred = []
		pred = False
		while True:
			line = popen.stdout.readline()
			if not line:
				break

			if pred:
				y_i = int(line)
				ypred.append(2*y_i - 1) #make sure it is -1/+1
			else:
				print ">>> " + line.rstrip()			
			
			if line.startswith("ypred"):
				pred = True

		popen.wait()
		#convert from single-instance to bag predictions:
		ypred = [max(ypred[k:k+len(X[k])]) for k in xrange(len(X))]
		assert(len(ypred) == len(X))
		return ypred