# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:02:08 2016

@author: snoran
"""

import os
import json
import numpy as np
search = "Best params:"
l = len(search)
best_class_weights = [[]]*20
for root, dirs, files in os.walk("."):
	for f in files:
		if f.endswith(".txt") and f.startswith("log"):
			print(os.path.join(root, f))
			for s in f[f.index("_"):f.index(".txt")].split("_"):
				if len(s) > 0:
					if s[0] == "p":
						participant = int(s[1:])
			with open(os.path.join(root, f), 'rb') as fd:
				text = fd.read()
			try:
				i=text.index(search)
				j=text[i:].index("}")
				best_params_str = text[i+l:i+j+2] #+2 because of inner {} pair
				print best_params_str
				best_params = json.loads(best_params_str.replace('-1:', '"-1":').replace('1:', '"1":').replace("'",'"'))
				print (best_params['class_weight'])
				best_class_weights[participant].append((best_params['class_weight']['1'], best_params['class_weight']['-1']))
				print ("participant %d" %participant)
			except:
				pass
		elif f.endswith("log.txt") and f.startswith("lopo"):
			print(os.path.join(root, f))
			for s in f[f.index("_"):f.index("log.txt")].split("_"):
				if len(s) > 0:
					if s[0] == "p":
						participant = int(s[1:])
			with open(os.path.join(root, f), 'rb') as fd:
				text = fd.read()
			try:
				i=text.index(search)
				j=text[i:].index("}")
				best_params_str = text[i+l:i+j+2] #+2 because of inner {} pair
				print best_params_str
				best_params = json.loads(best_params_str.replace('-1:', '"-1":').replace('1:', '"1":').replace("'",'"'))
				print (best_params['class_weight'])
				best_class_weights[participant].append((best_params['class_weight']['1'], best_params['class_weight']['-1']))
				print ("participant %d" %participant)
			except:
				pass
			
	
print(len(best_class_weights))
print(len(best_class_weights[0]))
print('\n\n\n')
print [np.mean(best_class_weights[k], axis=0) for k in range(len(best_class_weights))]
print [np.std(best_class_weights[k], axis=0) for k in range(len(best_class_weights))]