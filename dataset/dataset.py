'''
Function:
	load data for training.
Author:
	Charles
'''
import os
import cv2
import math
import numpy as np


'''
Function:
	For loading data of training.
Options:
	Please see config.py to learn about the meaning of options.
'''
class DataLoader():
	def __init__(self, **kwargs):
		self.options = kwargs
		self.initialize()
	'''initialize'''
	def initialize(self):
		self.trainSet = self.options.get('trainSet')
		with open(self.trainSet, 'r') as f:
			self.trainList = f.read().split('\n')
		np.random.shuffle(self.trainList)
		self.gen_scale = self.options.get('gen_scale')
		self.pic_dim = self.options.get('pic_dim')
		self.batch_size = self.options.get('batch_size')
		self.num_batches = math.ceil(len(self.trainList) / self.batch_size)
		self.video_shape = [self.gen_scale[1], self.gen_scale[0], self.gen_scale[0], self.pic_dim]
		self.num_frames = self.video_shape[0]
		self.netInput_size = self.gen_scale[0]
		self.imgSize = self.options.get('imgSize')
		self.rootDir = self.options.get('rootDir')
	'''get a batch data'''
	def iteration(self):
		for nb in range(self.num_batches):
			idx_start = nb * self.batch_size
			idx_end = min(len(self.trainList)-1, (nb+1)*self.batch_size)
			paths_batch = self.trainList[idx_start: idx_end]
			data_batch = np.zeros(shape=[self.batch_size] + self.video_shape)
			for video_idx, video_path in enumerate(paths_batch):
				imgs = cv2.imread(os.path.join(self.rootDir, video_path))
				data = np.zeros(shape=self.video_shape)
				num_imgs = imgs.shape[0] / self.imgSize
				for i in range(self.num_frames):
					if i < num_imgs:
						# Use only 1st 32 frames.
						pointer = int(i * self.imgSize)
					else:
						# Copy the last frame till num_frames.
						pointer = int((num_imgs - 1) * self.imgSize)
					img = imgs[pointer: pointer+self.imgSize, :]
					img_resize = cv2.resize(img, (self.netInput_size, self.netInput_size))
					img_norm = self.normalization(img_resize)
					data[i] = img_norm
				data_batch[video_idx] = data
			yield data_batch
	'''normalization, convert x to [-1, 1]'''
	def normalization(self, x):
		return cv2.normalize(x, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	'''return the number of batch for each epoch'''
	def __len__(self):
		return self.num_batches