# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:21:18 2016

@author: snoran
"""

from __future__ import division

import os
from argparse import ArgumentParser
import pickle
import numpy as np
from matplotlib import pyplot as plt
from util import accuracy_precision_recall_fscore

working_dir = "eval"

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
	
files = [f for f in os.listdir(res_dir) if os.path.isfile(os.path.join(res_dir,f)) and f.startswith("lopo_") and f.endswith(".pickle")]

confusion_matrix = {}
fscores = {}
for f in files:
	_p_index = f.index("_p")
	_b_index = f.index("_b")
	_m_index = f.index("_m")
	dot_index = f.index(".")
	#_i_index = f.index("_i")
	p = int(f[_p_index+2:_b_index])
	b = int(f[_b_index+2:_m_index])
	m = int(f[_m_index+2:dot_index])
	
	print p,b,m
	
	os.rename(os.path.join(res_dir,f), os.path.join(res_dir,"lopo_p%d_m%d_b%d_i0.pickle" %(p,m,b)))