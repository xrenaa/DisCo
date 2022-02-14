import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

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

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.contiguous().view(self.size)

# this encoder is for 64*64 resolution
class Baseline_Encoder(nn.Module):
    def __init__(self, z_dim=10, nc=1):
        super(Baseline_Encoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

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
            View((-1, 4096)),                   # B, 512
            nn.Linear(4096, 256),                # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim),               # B, z_dim
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        rep = self._encode(x)
        return rep

    def _encode(self, x):
        return self.encoder(x)
    
# this encoder is for 32*32 resolution
class Baseline_Encoder_1(nn.Module):
    def __init__(self, z_dim=10, nc=1):
        super(Baseline_Encoder_1, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          
            nn.LeakyReLU(),
            View((-1, 4096)),                   
            nn.Linear(4096, 256),                
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 
            nn.LeakyReLU(),
            nn.Linear(256, z_dim),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        rep = self._encode(x)
        return rep

    def _encode(self, x):
        return self.encoder(x)