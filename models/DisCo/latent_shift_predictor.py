import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torchvision.models import resnet18
import numpy as np

def save_hook(module, input, output):
    setattr(module, 'output', output)

class LatentShiftPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictor, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()