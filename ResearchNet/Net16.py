import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels, 7 output channels ( conv filters ), conv size (4x4)
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(3, 7, 5, padding=2)
        self.conv2 = nn.Conv2d(7, 7, 5, padding=2)
        self.conv3 = nn.Conv2d(7, 7, 5, padding=2)
    
        # 2 fully connected layers
        self.fc1 = nn.Linear(7*5*5, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # Apply max pooling after each convolution
        x = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), 2)
        x = F.max_pool2d(torch.tanh(self.conv3(x)), 2)
    
        # create a view of the data that flattens it
        x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        # reduce the last dimension
        x = F.log_softmax(self.fc2(x), dim=-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


