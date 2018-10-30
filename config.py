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
	-dis_dim: Scale for discriminator-channel.
	-gen_dim: Scale for generator-channel.
	-gen_scale: Scale for generator-width and height.
	-pic_dim: Dimension of image color.
	-noise_dim: Dimension of initial noise vector.
	-sample_size: Number of samples to be generated at once(for evaluate).
	-mask_L1_lambda: Weight for L1 regularizer of mask.
	-trainlogfile: Record the train info(filename).
	-modelSaved: Save the trained model(path).
	-samplesSaved: Save the sample videos from generator(path).
	-max_epoch: Number of training epochs.
	-save_interval: Save and test the model each save_interval epochs.
	-trainSet: The paths of the videos for training.
	-imgSize: The size of img(each frame) in the videos(trainSet).
	-root: Root path(the absolute path of training data folder).
'''
options = {
			'info': 'videoGan options',
			'batch_size': 32,
			'lr_g': 1e-4,
			'lr_d': 1e-5,
			'beta1_g': 0.5,
			'beta1_d': 0.5,
			'dis_dim': 64,
			'gen_dim': 64,
			'gen_scale': [64, 32, 16, 8, 4, 2],
			'pic_dim': 3,
			'noise_dim': 100,
			'sample_size': 32,
			'mask_L1_lambda': 0.1,
			'trainlogfile': 'train.log',
			'modelSaved': './modelSaved',
			'samplesSaved': './samplesSaved',
			'max_epoch': 1000,
			'save_interval': 1,
			'trainSet': 'golf.txt',
			'imgSize': 128,
			'rootDir': '/data1/zcjin/frames-stable-many'
			}