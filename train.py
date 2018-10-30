'''
Function:
	for model training.
Author:
	Charles
'''
import config
from nets.videoGan import videoGan
from dataset.dataset import DataLoader


'''main function.'''
def main():
	model = videoGan(trainlogfile=config.options.get('trainlogfile'),
					 modelSaved=config.options.get('modelSaved'),
					 samplesSaved=config.options.get('samplesSaved'),
					 batch_size=config.options.get('batch_size'),
					 dis_dim=config.options.get('dis_dim'),
					 gen_dim=config.options.get('gen_dim'),
					 gen_scale=config.options.get('gen_scale'),
					 pic_dim=config.options.get('pic_dim'),
					 lr_g=config.options.get('lr_g'),
					 lr_d=config.options.get('lr_d'),
					 beta1_g=config.options.get('beta1_g'),
					 beta1_d=config.options.get('beta1_d'),
					 noise_dim=config.options.get('noise_dim'),
					 sample_size=config.options.get('sample_size'),
					 mask_L1_lambda=config.options.get('mask_L1_lambda'),
					 max_epoch=config.options.get('max_epoch'),
					 save_interval=config.options.get('save_interval'))
	dl = DataLoader(trainSet=config.options.get('trainSet'),
					gen_scale=config.options.get('gen_scale'),
					pic_dim=config.options.get('pic_dim'),
					batch_size=config.options.get('batch_size'),
					imgSize=config.options.get('imgSize'),
					rootDir=config.options.get('rootDir'))
	model.train(dl)



if __name__ == '__main__':
	main()