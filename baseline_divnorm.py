"""BaselineDivNorm CNNs"""

import math
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from numpy.core.numeric import True_
from .divisive_norm_exc_inh import *

class BaselineDivNorm(nn.Module):
    def __init__(self, divnorm=True):
        super(BaselineDivNorm, self).__init__()

        self.divnorm = divnorm
        self.inplanes = 64
        self.batch_size = 1024

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, padding=6, kernel_size=13)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        if self.divnorm: self.div1 = DivNormExcInh(self.inplanes)
        self.relu1 = nn.ReLU(inplace=False)
       
        self.conv2 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, padding=3, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        if self.divnorm: self.div2 = DivNormExcInh(self.inplanes)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, padding=1, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(self.inplanes)
        if self.divnorm: self.div3 = DivNormExcInh(self.inplanes)
        self.relu3 = nn.ReLU(inplace=False)
        
        self.aap = nn.AdaptiveAvgPool2d((1,1))  # 1024 * 64 * 1 * 1
        
        self.fc1 = nn.Linear(in_features=self.inplanes, out_features=self.inplanes) 
        self.fc2 = nn.Linear(in_features=self.inplanes, out_features=2)
 
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        if self.divnorm: x = self.div1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.divnorm: x = self.div2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.divnorm: x = self.div3(x)
        x = self.relu3(x)

        x = self.aap(x)
        x = torch.squeeze(torch.squeeze(x, -1), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

#EOF
