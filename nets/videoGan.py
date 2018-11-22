'''
Function:
	videoGan model.
Author:
	Charles
'''
import os
import cv2
import time
import imageio
import numpy as np
import tensorflow as tf


'''
Function:
	Build the model of videoGan.
Options:
	Please see config.py to learn about the meaning of options.
'''
class videoGan():
	def __init__(self, **kwargs):
		self.options = kwargs
		self.initialize()
	'''initialize'''
	def initialize(self):
		# for record.
		self.logfile = open(self.options.get('trainlogfile'), 'w')
		self.modelSaved = self.options.get('modelSaved')
		self.samplesSaved = self.options.get('samplesSaved')
		# hyperparameter.
		self.batch_size = self.options.get('batch_size')
		self.dis_dim = self.options.get('dis_dim')
		self.gen_dim = self.options.get('gen_dim')
		self.gen_scale = self.options.get('gen_scale')
		self.pic_dim = self.options.get('pic_dim')
		self.lr_g = self.options.get('lr_g')
		self.lr_d = self.options.get('lr_d')
		self.beta1_g = self.options.get('beta1_g')
		self.beta1_d = self.options.get('beta1_d')
		self.noise_dim = self.options.get('noise_dim')
		self.sample_size = self.options.get('sample_size')
		self.mask_L1_lambda = self.options.get('mask_L1_lambda')
		self.max_epoch = self.options.get('max_epoch')
		self.save_interval = self.options.get('save_interval')
		self.video_shape = [self.gen_scale[1], self.gen_scale[0], self.gen_scale[0], self.pic_dim]
	'''call the function at the end of training'''
	def closure(self):
		self.logfile.close()
	'''call the function to train the model'''
	def train(self, dataloader):
		# define the placeholders.
		videos_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size]+self.video_shape, name='real_videos')
		noise_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dim], name='noise')
		# get the generator and discriminator.
		generator, mask_L1_penalty = self.gNet(noise_ph)
		dis_real, dis_real_logits = self.dNet(videos_ph)
		gen_samples = self.gNet(noise_ph, is_training=False, reuse=True)
		dis_fake, dis_fake_logits = self.dNet(generator, reuse=True)
		# calculate the loss.
		dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits, labels=tf.ones_like(dis_real)))
		dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, labels=tf.zeros_like(dis_fake)))
		dis_loss = dis_loss_real + dis_loss_fake
		gen_loss_no_penalty = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, labels=tf.ones_like(dis_fake)))
		gen_loss = self.mask_L1_lambda*mask_L1_penalty + gen_loss_no_penalty
		# distinguish the variable.
		trainable_vars = tf.trainable_variables()
		dis_vars = [var for var in trainable_vars if 'dNet_' in var.name]
		gen_vars = [var for var in trainable_vars if 'gNet_' in var.name]
		# calculate the number of parameters.
		total_parameters = 0
		for var in trainable_vars:
			params = 1
			for dim in var.get_shape().as_list():
				params *= int(dim)
				total_parameters += params
		loginfo = self.logger('<Total Parameters>: %d' % total_parameters)
		self.logfile.write(loginfo + '\n')
		# create optimizer.
		dis_optim = tf.train.AdamOptimizer(self.lr_d, beta1=self.beta1_d).minimize(dis_loss, var_list=dis_vars)
		gen_optim = tf.train.AdamOptimizer(self.lr_g, beta1=self.beta1_g).minimize(gen_loss, var_list=gen_vars)
		# training.
		tmp = set(tf.global_variables())
		self.saver = tf.train.Saver()
		with tf.Session() as session:
			self.session = session
			epoch_now = self.loadModel(ckptpath=self.modelSaved)
			if epoch_now:
				self.session.run(tf.variables_initializer(set(tf.global_variables()) - tmp))
			else:
				tf.global_variables_initializer().run()
				epoch_now = 0
			for epoch in range(epoch_now, self.max_epoch):
				dataloader.initialize()
				batchnum_per_epoch = len(dataloader)
				for batch_idx, data in enumerate(dataloader.iteration()):
					batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.noise_dim]).astype(np.float32)
					self.session.run([dis_optim], feed_dict={videos_ph: data, noise_ph: batch_noise})
					self.session.run([gen_optim], feed_dict={noise_ph: batch_noise})
					errorD_fake = dis_loss_fake.eval({noise_ph: batch_noise})
					errorD_real = dis_loss_real.eval({videos_ph: data})
					errorG = gen_loss.eval({noise_ph: batch_noise})
					loginfo = self.logger('[Epoch(%d/%d)-Batch(%d/%d)]: \n<errorD_fake>: %f, <errorD_real>: %f, <errorG>: %f' % (epoch, self.max_epoch, batch_idx, batchnum_per_epoch, errorD_fake, errorD_real, errorG))
					self.logfile.write(loginfo + '\n')
				if (epoch % self.save_interval == 0) and (epoch != 0):
					self.saveModel(savepath=self.modelSaved, epoch=epoch)
					sample_noise = np.random.uniform(-1, 1, [self.sample_size, self.noise_dim]).astype(np.float32)
					sample_videos = self.session.run([gen_samples], feed_dict={noise_ph: sample_noise})
					loginfo = self.logger('[Sample]: \nGenerate samples after epoch %d while training model...' % epoch)
					self.saveSamples(sample_videos, epoch, self.samplesSaved)
		self.closure()
	'''
	Function:
		discriminator
	Input:
		-x: the video to be discriminated.
		-reuse: whether share the variable or not.
		-is_training: training or test.
	'''
	def dNet(self, x, is_training=True, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			conv1, w_conv1, b_conv1 = self.conv3D(x, self.dis_dim, name='dNet_conv1', with_wb=True)
			conv1 = self.leakyRelu(conv1)
			conv2, w_conv2, b_conv2 = self.conv3D(conv1, self.dis_dim*2, name='dNet_conv2', with_wb=True)
			bn2 = self.leakyRelu(self.batchNorm(conv2, is_training=is_training, name='dNet_bn2'))
			conv3, w_conv3, b_conv3 = self.conv3D(bn2, self.dis_dim*4, name='dNet_conv3', with_wb=True)
			bn3 = self.leakyRelu(self.batchNorm(conv3, is_training=is_training, name='dNet_bn3'))
			conv4, w_conv4, b_conv4 = self.conv3D(bn3, self.dis_dim*8, name='dNet_conv4', with_wb=True)
			bn4 = self.leakyRelu(self.batchNorm(conv4, is_training=is_training, name='dNet_bn4'))
			fc, w_fc, b_fc = self.linear(tf.reshape(bn4, [self.batch_size, -1]), 1, name='dNet_fc', with_wb=True)
			return tf.nn.sigmoid(fc), fc
	'''
	Function:
		generator
	Input:
		-x: noise(100 dim).
		-reuse: whether share the variable or not.
		-is_training: training or test.
	'''
	def gNet(self, x, reuse=False, is_training=True):
		'''background stream'''
		with tf.variable_scope("generator") as scope:
			if reuse:
				scope.reuse_variables()
			bg_stream_fc, w_bg_stream_fc, b_bg_stream_fc = self.linear(x, self.gen_dim*8*self.gen_scale[4]*self.gen_scale[4], name='gNet_bg_stream_fc', with_wb=True)
			bg_stream_x = tf.reshape(bg_stream_fc, [-1, self.gen_scale[4], self.gen_scale[4], self.gen_dim*8])
			bg_stream_bn0 = tf.nn.relu(self.batchNorm(bg_stream_x, is_training=is_training, name='gNet_bg_stream_bn0'))
			bg_stream_deconv1, w_bg_stream_deconv1, b_bg_stream_deconv1 = self.deconv2D(bg_stream_bn0, [self.batch_size, self.gen_scale[3], self.gen_scale[3], self.gen_dim*4], name='gNet_bg_stream_deconv1', with_wb=True)
			bg_stream_bn1 = tf.nn.relu(self.batchNorm(bg_stream_deconv1, is_training=is_training, name='gNet_bg_stream_bn1'))
			bg_stream_deconv2, w_bg_stream_deconv2, b_bg_stream_deconv2 = self.deconv2D(bg_stream_bn1, [self.batch_size, self.gen_scale[2], self.gen_scale[2], self.gen_dim*2], name='gNet_bg_stream_deconv2', with_wb=True)
			bg_stream_bn2 = tf.nn.relu(self.batchNorm(bg_stream_deconv2, is_training=is_training, name='gNet_bg_stream_bn2'))
			bg_stream_deconv3, w_bg_stream_deconv3, b_bg_stream_deconv3 = self.deconv2D(bg_stream_bn2, [self.batch_size, self.gen_scale[1], self.gen_scale[1], self.gen_dim*1], name='gNet_bg_stream_deconv3', with_wb=True)
			bg_stream_bn3 = tf.nn.relu(self.batchNorm(bg_stream_deconv3, is_training=is_training, name='gNet_bg_stream_bn3'))
			bg_stream_deconv4, w_bg_stream_deconv4, b_bg_stream_deconv4 = self.deconv2D(bg_stream_bn3, [self.batch_size, self.gen_scale[0], self.gen_scale[0], self.pic_dim], name='gNet_bg_stream_deconv4', with_wb=True)
			background = tf.nn.tanh(bg_stream_deconv4)
			background = tf.tile(tf.reshape(background, [self.batch_size, 1, self.gen_scale[0], self.gen_scale[0], self.pic_dim]), [1, self.gen_scale[1], 1, 1, 1])
			'''foreground stream'''
			fg_stream_fc, w_fg_stream_fc, b_fg_stream_fc = self.linear(x, self.gen_dim*8*self.gen_scale[5]*self.gen_scale[4]*self.gen_scale[4], name='gNet_fg_stream_fc', with_wb=True)
			fg_stream_x = tf.reshape(fg_stream_fc, [-1, self.gen_scale[5], self.gen_scale[4], self.gen_scale[4], self.gen_dim*8])
			fg_stream_bn0 = tf.nn.relu(self.batchNorm(fg_stream_x, is_training=is_training, name='gNet_fg_stream_bn0'))
			fg_stream_deconv1, w_fg_stream_deconv1, b_fg_stream_deconv1 = self.deconv3D(fg_stream_bn0, [self.batch_size, self.gen_scale[4], self.gen_scale[3], self.gen_scale[3], self.gen_dim*4], name='gNet_fg_stream_deconv1', with_wb=True)
			fg_stream_bn1 = tf.nn.relu(self.batchNorm(fg_stream_deconv1, is_training=is_training, name='gNet_fg_stream_bn1'))
			fg_stream_deconv2, w_fg_stream_deconv2, b_fg_stream_deconv2 = self.deconv3D(fg_stream_bn1, [self.batch_size, self.gen_scale[3], self.gen_scale[2], self.gen_scale[2], self.gen_dim*2], name='gNet_fg_stream_deconv2', with_wb=True)
			fg_stream_bn2 = tf.nn.relu(self.batchNorm(fg_stream_deconv2, is_training=is_training, name='gNet_fg_stream_bn2'))
			fg_stream_deconv3, w_fg_stream_deconv3, b_fg_stream_deconv3 = self.deconv3D(fg_stream_bn2, [self.batch_size, self.gen_scale[2], self.gen_scale[1], self.gen_scale[1], self.gen_dim*1], name='gNet_fg_stream_deconv3', with_wb=True)
			fg_stream_bn3 = tf.nn.relu(self.batchNorm(fg_stream_deconv3, is_training=is_training, name='gNet_fg_stream_bn3'))
			# get the mask.
			mask, mask_w, mask_b = self.deconv3D(fg_stream_bn3, [self.batch_size, self.gen_scale[1], self.gen_scale[0], self.gen_scale[0], 1], with_wb=True, name='gNet_mask')
			mask = tf.nn.sigmoid(mask)
			# get the foreground.
			fg_stream_deconv4, w_fg_stream_deconv4, w_fg_stream_deconv4 = self.deconv3D(fg_stream_bn3, [self.batch_size, self.gen_scale[1], self.gen_scale[0], self.gen_scale[0], self.pic_dim], name='gNet_fg_stream_deconv4', with_wb=True)
			foreground = tf.nn.tanh(fg_stream_deconv4)
			'''merge the foreground and background'''
			foreground_masked = tf.multiply(foreground, mask)
			background_masked = tf.multiply(background, tf.subtract(tf.constant([1.0]), mask))
			video = tf.add(foreground_masked, background_masked)
			# for L1 regularizer penalty of mask.
			return (video, tf.reduce_mean(tf.reduce_sum(tf.abs(mask_w)))) if is_training else video
	'''2D convolution'''
	def conv2D(self,
			   x,
			   out_channels,
			   filter_h=4,
			   filter_w=4,
			   stride_h=2,
			   stride_w=2,
			   stddev=0.02,
			   with_wb=False,
			   name='conv2d'):
		with tf.variable_scope(name):
			w = tf.get_variable('weight', [filter_h, filter_w, x.get_shape()[-1], out_channels], initializer=tf.truncated_normal_initializer(stddev=stddev))
			b = tf.get_variable('bias', [out_channels], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
			conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
			return (conv, w, b) if with_wb else conv
	'''3D convolution'''
	def conv3D(self, 
			   x, 
			   out_channels,
			   filter_d=4,
			   filter_h=4,
			   filter_w=4,
			   stride_d=2,
			   stride_h=2,
			   stride_w=2,
			   stddev=0.02,
			   with_wb=False,
			   name='conv3d'):
		with tf.variable_scope(name):
			w = tf.get_variable('weight', [filter_d, filter_h, filter_w, x.get_shape()[-1], out_channels], initializer=tf.truncated_normal_initializer(stddev=stddev))
			b = tf.get_variable('bias', [out_channels], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.conv3d(x, w, strides=[1, stride_d, stride_h, stride_w, 1], padding='SAME')
			conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
			return (conv, w, b) if with_wb else conv
	'''2D transposed convolution'''
	def deconv2D(self, 
				 x,
				 output_shape,
				 filter_h=4,
				 filter_w=4,
				 stride_h=2,
				 stride_w=2,
				 stddev=0.02,
				 with_wb=False,
				 name='deconv2d'):
		with tf.variable_scope(name):
			# [height, width, out_channels, in_channels]
			w = tf.get_variable('weight', [filter_h, filter_w, output_shape[-1], x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
			b = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
			deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride_h, stride_w, 1])
			deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
			return (deconv, w, b) if with_wb is True else deconv
	'''3D transposed convolution'''
	def deconv3D(self,
				 x,
				 output_shape,
				 filter_d=4,
				 filter_h=4,
				 filter_w=4,
				 stride_d=2,
				 stride_h=2,
				 stride_w=2,
				 stddev=0.02,
				 with_wb=False,
				 name='deconv3d'):
		with tf.variable_scope(name):
			# [time, height, width, out_channels, in_channels]
			w = tf.get_variable('weight', [filter_d, filter_h, filter_w, output_shape[-1], x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
			b = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
			deconv = tf.nn.conv3d_transpose(x, w, output_shape=output_shape, strides=[1, stride_d, stride_h, stride_w, 1])
			deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
			return (deconv, w, b) if with_wb else deconv
	'''batch normalization'''
	def batchNorm(self, x, eps=1e-5, decay=0.9, is_training=True, name='batch_norm'):
		with tf.variable_scope(name):
			bn = tf.contrib.layers.batch_norm(x,
											  decay=decay,
											  updates_collections=None,
											  epsilon=eps,
											  scale=True,
											  is_training=is_training)
			return bn
	'''fully connected'''
	def linear(self, x, output_size, stddev=0.02, with_wb=False, name='linear'):
		with tf.variable_scope(name):
			w = tf.get_variable('weight', [x.get_shape()[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
			b = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.0))
			fc = tf.matmul(x, w) + b
			return (fc, w, b) if with_wb else fc
	'''leaky relu'''
	def leakyRelu(self, x, leaky=0.2):
		return tf.maximum(x, x * leaky)
	'''print the info with time'''
	def logger(self, message):
		info = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
		print(info)
		return info
	'''save the checkpoints'''
	def saveModel(self, savepath, epoch):
		if not os.path.exists(savepath):
			os.mkdir(savepath)
		epoch_dir = os.path.join(savepath, str(epoch))
		if not os.path.exists(epoch_dir):
			os.mkdir(epoch_dir)
		self.saver.save(self.session, os.path.join(epoch_dir, 'model.ckpt'))
		self.logger('Checkpoint of %d saved into %s...' % (epoch, epoch_dir))
	'''load the checkpoints'''
	def loadModel(self, ckptpath):
		if not os.path.exists(ckptpath):
			self.logger('No checkpoints found, start to train a new model...')
			return False
		ckptdirs = os.listdir(ckptpath)
		if len(ckptdirs) < 1:
			self.logger('No checkpoints found, start to train a new model...')
			return False
		ckptdirs = [int(i) for i in ckptdirs]
		ckptfile = os.path.join(ckptpath, str(max(ckptdirs)), 'model.ckpt')
		self.saver.restore(self.session, ckptfile)
		self.logger('Checkpoint of %s looded successfully...' % ckptfile)
		return max(ckptdirs)
	'''Save sample videos from generator'''
	def saveSamples(self, samples, epoch, savepath):
		if not os.path.exists(savepath):
			os.mkdir(savepath)
		epoch_dir = os.path.join(savepath, 'epoch_'+str(epoch))
		if not os.path.exists(epoch_dir):
			os.mkdir(epoch_dir)
		samples = cv2.normalize(np.array(samples), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		samples = np.reshape(samples, [samples.shape[1], samples.shape[2], samples.shape[3], samples.shape[4], samples.shape[5]])
		for idx, sample in enumerate(samples):
			frames = []
			for i in range(sample.shape[0]):
				frames += [sample[i]]
			imageio.mimsave(os.path.join(epoch_dir, 'sample_%d.gif' % idx), frames)