from ResearchNet.Augmentations.Vae import VAE, loss_function
import torch
import torch.utils.data
from torch import nn, optim
import numpy as np

def back_to_image(image):
	return np.resize(np.uint8(image.detach().cpu()*255), (102, 102, 3))

def train_network(epochs, model, optimizer, datas):
	model.train()
	train_loss = 0
	for i in range(epochs):
		for batch_idx, data in enumerate(datas):
			optimizer.zero_grad()
			recon_batch, mu, logvar = model(data)
			loss = loss_function(recon_batch, data, mu, logvar)
			loss.backward()
			train_loss += loss.item()
			optimizer.step()
			if batch_idx % 10 == 0:
				print('Train epoch {}, \t{:.0f}%'.format(i, loss.item()/len(data)))	
	pass

class JustEncode:
	def __init__(self, training_set, classes=[], rate=1, torch_device=torch.device('cuda')):
		self.classes = classes
		self.network = VAE()
		self.network.cuda()
		self.training_set = torch.utils.data.DataLoader(training_set, batch_size=100)
		self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
		self.rate = rate
		train_network(1000, self.network, self.optimizer, self.training_set)
		self.device = torch_device

	def __call__(self, image_label):
		image, label = image_label
		images = []
		labels = []
		if label in self.classes:
			augment_with = torch.from_numpy(np.float32(image/255)).to(self.device)
			for transform in range(self.rate):
				new_image, mu, std = self.network(augment_with)
				images.append(back_to_image(new_image))
				labels.append(label)
		return (images, labels)

		
		

class EncodeOpposite:
	def __init__(self, training_sets, classes=[], rate=1, torch_device=torch.device('cuda')):
		self.classes = classes
		self.networks = [VAE() for x in classes]
		# move all of the networks to the cuda
		for network in self.networks:
			network.cuda()
		self.optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in self.networks]
		self.datasets = list(map(lambda x :torch.utils.data.DataLoader( x, batch_size=100 ), training_sets))
		self.rate = rate
		self.label_for_label = lambda x : 0 if x == 1 else 1
		self.perform_training()
		self.device = torch_device

	def perform_training(self):
		for i, network in enumerate(self.networks):
			# train the network
			train_network(1000, network, self.optimizers[i], self.datasets[i])
	
	def __call__(self, image_label):
		image, label = image_label
		images = []
		labels = []
		if label in self.classes:
			append_label = self.label_for_label(label)
			augment_with = torch.from_numpy(np.float32(image/255)).to(self.device)
			for transform in range(self.rate):
				new_image, mu, std = self.networks[append_label](augment_with)
				images.append(back_to_image(new_image))
				labels.append(append_label)
		return (images, labels)

class EncodeSame(EncodeOpposite):
	def __init__(self, training_sets, classes=[], rate=1, torch_device=torch.device('cuda')):
		super().__init__(training_sets, classes=classes, rate=rate, torch_device=torch_device)
		# use the same class to train as to encode when augmenting
		self.label_for_label = lambda x : x
		
class DecodeRandom(EncodeSame):
	def __init__(self, training_sets, classes=[], rate=1, torch_device=torch.device('cuda')):
		super().__init__(training_sets, classes=classes, rate=rate, torch_device=torch_device)
		images_0, mu_0, std_0 = self.networks[0](training_sets[0])
		images_1, mu_1, std_1 = self.networks[1](training_sets[1])
		mus = map(lambda x : np.float32(x.detach().cpu()), (mu_0, mu_1))
		stds = map(lambda x : np.float32(x.detach().cpu()), (std_0, std_1))
		self.mus = [torch.from_numpy(np.mean(mu, axis=0)).to(torch_device) for mu in mus]
		self.stds = [torch.from_numpy(np.mean(std, axis=0)).to(torch_device) for std in stds]

	def __call__(self, image_label):
		image, label = image_label
		images = []
		labels = []
		if label in self.classes:
			append_label = self.label_for_label(label)
			augment_with = torch.from_numpy(np.float32(image/255)).to(self.device)
			for transform in range(self.rate):
				new_image = self.networks[append_label].decode(torch.normal(self.mus[append_label], self.stds[append_label]))
				images.append(back_to_image(new_image))
				labels.append(append_label)
		return (images, labels)
