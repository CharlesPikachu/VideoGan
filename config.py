'''
Function:
	Settings for train and test.
Author:
	Charles
'''


'''
options explain:
	-info: Introduce options.
	-batch_size: Number of videos to process in a batch.
	-lr_g: Learning rate for generator.
	-lr_d: Learning rate for discriminator.
	-beta1_g: Beta1 for generator.
	-beta1_d: Beta1 for discriminator.
	-video_shape: The shape of a video.
	-dis_dim: Scale for discriminator-channel.
	-gen_dim: Scale for generator-channel.
	-gen_scale: Scale for generator-width and height.
	-pic_dim: Dimension of image color.
	-noise_dim: Dimension of initial noise vector.
'''
options = {
			'info': 'videoGan options',
			'batch_size': 32,
			'lr_g': 1e-4,
			'lr_d': 1e-5,
			'beta1_g': 0.5,
			'beta1_d': 0.5,
			'video_shape': [32, 64, 64, 3],
			'dis_dim': 64,
			'gen_dim': 64,
			'gen_scale': [64, 32, 16, 8, 4, 2],
			'pic_dim': 3,
			'noise_dim': 100
			}