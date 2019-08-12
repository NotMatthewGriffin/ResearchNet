from ResearchNet.DataLoader import DataLoader
from ResearchNet.IndicesDecider import IndicesDecider
import numpy as np
import torch

class AutoEncoderLoader(DataLoader):
	def __init__(self, dataset, random_seed, folds):
		DataLoader.__init__(self, dataset)
		self.indices_decider = IndicesDecider(len(self.files_to_read), folds, random_seed)

	def load_data(self, n, starting_size=(102, 102), ending_size=(102, 102), torch_device=torch.device('cuda')):
		indices_to_use = self.indices_decider.get_indices(n)['training']
		all_data = DataLoader.load_data(self, indices_to_use, starting_size=starting_size, ending_size=ending_size)
		classes_data = [[], []]
		for i, label in enumerate(all_data['labels']):
			classes_data[label].append(all_data['images'][i])
		data = list(map(lambda x : torch.from_numpy(np.resize(np.float32(x), (-1, 102*102*3))/255).to(torch_device), classes_data))
		return data

	def load_data_together(self, n, starting_size=(102, 102), ending_size=(102, 102), torch_device=torch.device('cuda')):
		indices_to_use = self.indices_decider.get_indices(n)['training']
		all_data = DataLoader.load_data(self, indices_to_use, starting_size=starting_size, ending_size=ending_size)
		data = torch.from_numpy(np.resize(np.float32(all_data['images']), (-1, 102*102*3))/255).to(torch_device)
		return data
