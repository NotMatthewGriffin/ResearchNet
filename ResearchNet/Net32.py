import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	
	def __init__(self):
		super(Net, self).__init__()
		# 4 convolutional layers
		# 3 input image channels, 7 output filters 5 x 5 kernel
		self.conv1 = nn.Conv2d(3, 7, 5, padding=10)
		self.conv2 = nn.Conv2d(7, 14, 5, padding=0)
		self.conv3 = nn.Conv2d(14, 21, 5, padding=0)
		self.conv4 = nn.Conv2d(21, 50, 2, padding=0)

		# fully connected output layer
		self.fcout = nn.Linear(50, 2)

	def forward(self, x):
		# Apply max pooling after convolution
		x = torch.tanh(self.conv1(x))
		x = F.max_pool2d(x,2)
		x = torch.tanh(self.conv2(x))
		x = F.max_pool2d(x,2)
		x = torch.tanh(self.conv3(x))
		x = F.max_pool2d(x,2)
		x = torch.tanh(self.conv4(x))
		x = F.max_pool2d(x,3)
		
		# flatten data
		x = x.view(-1, self.num_flat_features(x))
		x = F.log_softmax(self.fcout(x), dim=-1)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
