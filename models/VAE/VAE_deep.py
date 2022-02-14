import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.contiguous().view(self.size)

class GANbaseline(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(GANbaseline, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            View((-1, 4096)),                  # B, 512
            nn.Linear(4096, 256),              # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 4096),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 7, 1, 3), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
        
class GANbaseline2(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(GANbaseline2, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            View((-1, 4096)),                  # B, 512
            nn.Linear(4096, 256),              # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 4096),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 7, 1, 3), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def zcomplex(self, z):
        real = torch.sin(2*np.pi*z/self.N)
        imag = torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)
    
    
class GANbaseline3(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(GANbaseline3, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            View((-1, 4096)),                  # B, 512
            nn.Linear(4096, 256),              # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 4096),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 7, 1, 3), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()