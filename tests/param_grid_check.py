# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:02:08 2016

@author: snoran
"""

import os
import pickle
for root, dirs, files in os.walk("."):
	for f in files:
		if f.endswith(".pickle") and f.startswith("lopo"):
			for s in f[f.index("_"):f.index(".pickle")].split("_"):
				if len(s) > 0:
					if s[0] == "p":
						participant = int(s[1:])
			with open(os.path.join(root, f), 'rb') as fd:
				r = pickle.load(fd)
			k = r["Results"].keys()
			print(k)
			print ("participant %d" %participant)