from ResearchNet.Net32 import Net as Net32
from ResearchNet import DataLoader, KFoldSplitter, NNTrainer, AutoEncoderLoader
from ResearchNet.Augmentations import Translation, Rotation, Scale, Geometric, EncodeOpposite, EncodeSame, DecodeRandom, JustEncode, Duplicate
from ResearchNet.Preprocessors import Cropper
from ResearchNet.Reporters import report_recall
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import cv2
from sys import argv

data_set = '../newSelect/'
number_splits = 20
random_seed = 1

def main(n, name='', augmentations=[], worst_case_augmentations=[], no_cuda=False):
	print(n)
	file_name = '../augmentationResults/{}resFile.txt'.format(name)
	worst_case_file = '../augmentationResults/{}{}resFile.txt'.format('worstCase', name)
	
	device = torch.device('cpu') if no_cuda or not torch.cuda.is_available() else torch.device('cuda')
	net = Net32()
	if not no_cuda and torch.cuda.is_available():
		print('Using cuda(gpu) to train and process nn operations')
		net.cuda()
	dl = DataLoader(data_set)
	cropper = Cropper(top=31, left=31, width=40, height=40)
	splitter = KFoldSplitter(dl,
                                 k=number_splits, 
                                 seed=random_seed, 
                                 starting_size=(102, 102), 
                                 ending_size=(40, 40),
                                 augmentations=augmentations,
                                 worst_case_augmentations=worst_case_augmentations,
                                 cropper=cropper)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.RMSprop(net.parameters(), lr=0.0001)
	trainer = NNTrainer(net, optimizer, criterion)
	
	data = splitter.load_fold_n(n)
	for num, label in enumerate(['healthy', 'leuko']):
		print('Number of {} is {}'.format(label, np.sum(data['testing']['labels']==num)+np.sum(data['training']['labels']==num)))
	data = convert_data_to_tensor(data, device=device)

	# trained
	trainer.train_for_iterations(data['training'], iterations=300, batch_size=100)
	# report recall with unaugmented test set
	report_recall(net, data['testing'], file_name, n)
	# augment test set
	worst_case_data = convert_data_to_tensor(splitter.worst_case_test_data(n), device=device)
	# report recall with augmented test set
	report_recall(net, worst_case_data['testing'], worst_case_file, n)
	

def convert_data_to_tensor(data, device=torch.device('cpu')):
	for key in data.keys():
		print(key)
		images_to_fix = data[key]['images']
		labels_to_fix = data[key]['labels']
	
		labels_to_fix = torch.from_numpy(labels_to_fix).to(device)
	
		# move the channels before all but the batch dimension
		print(images_to_fix.shape)
		print(images_to_fix[0].shape)
		images_view = np.moveaxis(images_to_fix, -1, 1)
		data[key]['images'] = torch.from_numpy(images_view).to(device)
		data[key]['labels'] = labels_to_fix
	return data

def interpret_arguments(arguments):
	n = int(arguments[1])
	autoEncoderLoader = AutoEncoderLoader(data_set, random_seed, number_splits)
	expected_augmentations = {'-translate':lambda x: Translation(classes=[0, 1], rate=x, max_x=30, max_y=30),
                                  '-rotate':lambda x: Rotation(classes=[0, 1], rate=x, max_angle=180),
                                  '-scale':lambda x: Scale(classes=[0, 1], rate=x),
                                  '-geometric':lambda x: Geometric(classes=[0, 1], rate=x),
                                  '-encodeOpposite':lambda x: EncodeOpposite(autoEncoderLoader.load_data(n), classes=[0, 1], rate=x),
                                  '-encodeSame':lambda x : EncodeSame(autoEncoderLoader.load_data(n), classes=[0, 1], rate=x),
                                  '-decodeRandom':lambda x : DecodeRandom(autoEncoderLoader.load_data(n), classes=[0, 1], rate=x),
                                  '-duplicate':lambda x : Duplicate(classes=[0, 1], rate=x),
				  '-justEncode':lambda x : JustEncode(autoEncoderLoader.load_data_together(n), classes=[0, 1], rate=x),
				  '-smallTranslation':lambda x : Translation(classes=[0, 1], rate=x, max_x=10, max_y=10)
                                  }
	eworst_case_augmentations = {'-translate': lambda : Geometric(classes=[0,1], rate=10),
                                    '-rotate':lambda : Geometric(classes=[0,1], rate=10),
                                    '-scale':lambda : Geometric(classes=[0,1], rate=10),
                                    '-geometric':lambda : Geometric(classes=[0,1], rate=10),
                                    '-encodeOpposite': lambda : Geometric(classes=[0,1], rate=10),
                                    '-encodeSame': lambda : Geometric(classes=[0,1], rate=10),
                                    '-decodeRandom': lambda : Geometric(classes=[0,1], rate=10),
                                    '-duplicate': lambda : Geometric(classes=[0,1], rate=10),
                                    '-justEncode': lambda : Geometric(classes=[0,1], rate=10),
				    '-smallTranslation':lambda  : Geometric(classes=[0,1], rate=10)
                                    }
	augmentations = []
	augmentation_name = ''
	worst_case_augmentations = []
	for a_key in expected_augmentations:
		if a_key in arguments:
			rate_index = arguments.index(a_key)+1
			augmentations.append(expected_augmentations[a_key](int(arguments[rate_index])))
			worst_case_augmentations.append(eworst_case_augmentations[a_key]())
			augmentation_name += a_key[1:]+arguments[rate_index]
	return {'n':n, 'augmentations':augmentations, 'name':augmentation_name, 'no_cuda':False, 'worst_case_augmentations':worst_case_augmentations}
	

if __name__ == '__main__':
	main(**interpret_arguments(argv))
