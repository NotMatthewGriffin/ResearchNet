from torch import nn
import torch
from torch.nn import functional as F

flat_count = 102*102*3

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encoding layers
        self.fc1 = nn.Linear(flat_count, 400)
        self.fc21 = nn.Linear(400, 4)
        self.fc22 = nn.Linear(400, 4)
        # decoding layers
        self.fc3 = nn.Linear(4, 400)
        self.fc4 = nn.Linear(400, flat_count)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, flat_count))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# this is the loss fuction for vaes, it is the reconstruction error + KL divergence
# this is like a regulizer that keeps our generated encodings close to ~ N(0, 1)
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, flat_count), reduction='sum')
    KLD = -0.5 * torch.sum(1+logvar -mu.pow(2) - logvar.exp())
    return BCE + KLD
