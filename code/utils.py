# Written by: Erick Cobos T
# Date: October-2016
""" Some utility functions. """
import numpy as np
import time

def log(*messages):
	""" Simple logging function."""
	formatted_time = "[{}]".format(time.ctime())
	print(formatted_time, *messages)

def read_csv_info(csv_path):
	""" Reads the csv file and returns two lists: one with image filenames 
	(first column) and one with label filenames (second column)."""
	filenames = np.loadtxt(csv_path, dtype=bytes, delimiter=',').astype(str)
	
	return list(filenames[:,0]), list(filenames[:,1])
