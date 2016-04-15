# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:02:08 2016

@author: snoran
"""

import os
import json
search = "Best params:"
l = len(search)
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
				best_params = json.loads(best_params_str)
				print (best_params.keys())
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
				best_params = json.loads(best_params_str)
				print (best_params.keys())
				print ("participant %d" %participant)
			except:
				pass