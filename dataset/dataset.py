'''
Function:
	load data for training.
Author:
	Charles
'''
import math
import numpy as np


'''
Function:
	Dataset class for loading data of training.
Options:
	Please see config.py to learn about the meaning of options.
'''
class Dataset():
	def __init__(self, **kwargs):
		self.options = kwargs
		self.initialize()
	'''initialize'''
	def initialize(self):
		self.trainSet = self.options.get('trainSet')
		with open(self.trainSet, 'r') as f:
			self.trainList = f.read().split('\n')
		self.batch_size = self.options.get('batch_size')
		self.num_batches = math.ceil(len(self.trainList) / self.batch_size)
	''''''
	def iteration(self):
		for i in range(self.num_batches):
			
	'''return the number of batch for each epoch'''
	def __len__(self):
		return self.num_batches