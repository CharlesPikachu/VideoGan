'''
Function:
	videoGan model.
Author:
	Charles
'''
import os
import time
import tensorflow as tf


'''
Function:
	Build the model of videoGan.
Options:
	Please see config.py to understand the meaning of options.
'''
class videoGan():
	def __init__(self, **kwargs):
		self.options = kwargs
		self.initialize()
	'''initialize'''
	def initialize(self):
		self.saver = tf.train.Saver()
		self.batch_size = self.options.get('batch_size')
		self.video_shape = self.options.get('video_shape')
		self.dis_dim = self.options.get('dis_dim')
		self.gen_dim = self.options.get('gen_dim')
		self.gen_scale = self.options.get('gen_scale')
		self.pic_dim = self.options.get('pic_dim')
		self.lr_g = self.options.get('lr_g')
		self.lr_d = self.options.get('lr_d')
		self.beta1_g = self.options.get('beta1_g')
		self.beta1_d = self.options.get('beta1_d')
		self.noise_dim = self.options.get('noise_dim')

	'''call the function to train the model'''
	def train(self):
		
		d_optim = tf.train.AdamOptimizer(self.lr_d, beta1=beta1_d).minimize(self.d_loss, var_list=self.d_vars)
		pass

	'''
	Function:
		discriminator
	Input:
		-x: the video to be discriminated.
		-reuse: whether share the variable or not.
	'''
	def dNet(self, x, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		conv1 = self.leakyRelu(self.conv3D(x, self.dis_dim, name='dNet_conv1'))
		conv2 = self.conv3D(conv1, self.dis_dim*2, name='dNet_conv2')
		bn2 = self.leakyRelu(self.batchNorm(conv2, is_training=is_training, name='dNet_bn2'))
		conv3 = self.conv3D(bn2, self.dis_dim*4, name='dNet_conv3')
		bn3 = self.leakyRelu(self.batchNorm(conv3, is_training=is_training, name='dNet_bn3'))
		conv4 = self.conv3D(bn3, self.dis_dim*8, name='dNet_conv4')
		bn4 = self.leakyRelu(self.batchNorm(conv4, is_training=is_training, name='dNet_bn4'))
		fc = self.linear(tf.reshape(bn4, [self.batch_size, -1]), 1, name='dNet_fc')
		return fc, tf.nn.sigmoid(fc)
	'''
	Function:
		generator
	Input:
		-x: noise(100 dim).
		-reuse: whether share the variable or not.
		-is_training: training or test.
	'''
	def gNet(self, x, reuse=False, is_training=True):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		'''background stream'''
		bg_stream_fc = self.linear(x, self.gen_dim*8*self.gen_scale[4]*self.gen_scale[4], 'bg_stream_fc')
		bg_stream_x = tf.reshape(bg_stream_fc, [-1, self.gen_scale[4], self.gen_scale[4], self.gen_dim*8])
		bg_stream_bn0 = tf.nn.relu(self.batchNorm(bg_stream_x, is_training=is_training, name='bg_stream_bn0'))
		bg_stream_deconv1 = self.deconv2D(bg_stream_bn0, [self.batch_size, self.gen_scale[3], self.gen_scale[3], self.gen_dim*4], name='bg_stream_deconv1')
		bg_stream_bn1 = tf.nn.relu(self.batchNorm(bg_stream_deconv1, is_training=is_training, name='bg_stream_bn1'))
		bg_stream_deconv2 = self.deconv2D(bg_stream_bn1, [self.batch_size, self.gen_scale[2], self.gen_scale[2], self.gen_dim*2], name='bg_stream_deconv2')
		bg_stream_bn2 = tf.nn.relu(self.batchNorm(bg_stream_deconv2, is_training=is_training, name='bg_stream_bn2'))
		bg_stream_deconv3 = self.deconv2D(bg_stream_bn2, [self.batch_size, self.gen_scale[1], self.gen_scale[1], self.gen_dim*1], name='bg_stream_deconv3')
		bg_stream_bn3 = tf.nn.relu(self.batchNorm(bg_stream_deconv3, is_training=is_training, name='bg_stream_bn3'))
		bg_stream_deconv4 = self.deconv2D(bg_stream_bn3, [self.batch_size, self.gen_scale[0], self.gen_scale[0], self.pic_dim], name='bg_stream_deconv4')
		background = tf.nn.tanh(bg_stream_deconv4)
		background = tf.tile(tf.reshape(background, [self.batch_size, 1, self.gen_scale[0], self.gen_scale[0], self.pic_dim]), [1, self.gen_scale[1], 1, 1, 1])
		'''foreground stream'''
		fg_stream_fc = self.linear(x, self.gen_dim*8*self.gen_scale[5]*self.gen_scale[4]*self.gen_scale[4], 'fg_stream_fc')
		fg_stream_x = tf.reshape(fg_stream_fc, [-1, self.gen_scale[5], self.gen_scale[4], self.gen_scale[4], self.gen_dim*8])
		fg_stream_bn0 = tf.nn.relu(self.batchNorm(fg_stream_x, is_training=is_training, name='fg_stream_bn0'))
		fg_stream_deconv1 = self.deconv3D(fg_stream_bn0, [self.batch_size, self.gen_scale[4], self.gen_scale[3], self.gen_scale[3], self.gen_dim*4], name='fg_stream_deconv1')
		fg_stream_bn1 = tf.nn.relu(self.batchNorm(fg_stream_deconv1, is_training=is_training, name='fg_stream_bn1'))
		fg_stream_deconv2 = self.deconv3D(fg_stream_bn1, [self.batch_size, self.gen_scale[3], self.gen_scale[2], self.gen_scale[2], self.gen_dim*2], name='fg_stream_deconv2')
		fg_stream_bn2 = tf.nn.relu(self.batchNorm(fg_stream_deconv2, is_training=is_training, name='fg_stream_bn2'))
		fg_stream_deconv3 = self.deconv3D(fg_stream_bn2, [self.batch_size, self.gen_scale[2], self.gen_scale[1], self.gen_scale[1], self.gen_dim*1], name='fg_stream_deconv3')
		fg_stream_bn3 = tf.nn.relu(self.batchNorm(fg_stream_deconv3, is_training=is_training, name='fg_stream_bn3'))
		# get the mask
		mask, mask_w, mask_b = self.deconv3D(fg_stream_bn3, [self.batch_size, self.gen_scale[1], self.gen_scale[0], self.gen_scale[0], 1], with_wb=True, name='g_mask')
		mask = tf.nn.sigmoid(mask)
		# get the foreground
		fg_stream_deconv4 = self.deconv3D(fg_stream_bn3, [self.batch_size, self.gen_scale[1], self.gen_scale[0], self.gen_scale[0], self.pic_dim], name='fg_stream_deconv4')
		foreground = tf.nn.tanh(fg_stream_deconv4)
		'''merge the foreground and background'''
		foreground_masked = tf.mul(foreground, mask)
		background_masked = tf.mul(background, tf.sub(tf.constant([1.0]), mask))
		video = tf.add(foreground, background)
		# for L1 regularizer penalty
		return video, tf.reduce_mean(tf.reduce_sum(tf.abs(mask_w))) if is_training else video
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
			conv = tf.nn.conv3d(x, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
			conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
			return conv, w, b if with_wb else conv
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
			   std=0.02,
			   with_wb=False,
			   name='conv3d'):
		with tf.variable_scope(name):
			w = tf.get_variable('weight', [filter_d, filter_h, filter_w, x.get_shape()[-1], out_channels], initializer=tf.truncated_normal_initializer(stddev=stddev))
			b = tf.get_variable('bias', [out_channels], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.conv3d(x, w, strides=[1, stride_d, stride_h, stride_w, 1], padding='SAME')
			conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
			return conv, w, b if with_wb else conv
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
			deconv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())
			return deconv, w, b if with_wb else deconv
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
			deconv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())
			return deconv, w, b if with_wb else deconv
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
			return fc, w, b if with_wb else fc
	'''leaky relu'''
	def leakyRelu(self, leaky=0.2):
		return tf.maximum(x, x * leaky)
	'''print the info with time'''
	def logger(self, message):
		print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
	'''save the checkpoints.'''
	def saveModel(self, savepath, epoch, session):
		if not os.path.exists(savepath):
			os.mkdir(savepath)
		epoch_dir = os.path.join(savepath, str(epoch))
		if not os.path.exists(epoch_dir):
			os.mkdir(epoch_dir)
		self.saver.save(session, os.path.join(epoch_dir, 'model.ckpt'))
		self.logger('Checkpoint of %d saved into %s...' % (epoch, epoch_dir))