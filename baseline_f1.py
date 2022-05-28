import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from .divisive_norm_exc_inh import *

class BaselineF1(nn.Module):
    def __init__(self, divnorm=False, L3=False, L5=False, L7=False):
        super().__init__()

        self.divnorm = divnorm
        self.L3 = L3
        self.L5 = L5
        self.L7 = L7
        self.kernal_size = 5
        self.padding = 2
        self.inplanes = 32
        self.batch_size = 1024

        if self.L3:
            self.L5 = False
            self.L7 = False
        elif self.L5:
            self.L3 = True
            self.L7 = False
        elif self.L7:
            self.L3 = True
            self.L5 = True

        if self.L3:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding)
            if self.divnorm: self.div1 = DivNormExcInh(self.inplanes)
            self.conv2 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding)
            self.conv3 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding)
        
        if self.L5: 
            self.conv4 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                kernel_size=self.kernal_size, padding=self.padding)
            self.conv5 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                kernel_size=self.kernal_size, padding=self.padding)
        
        if self.L7:
            self.conv6 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                kernel_size=self.kernal_size, padding=self.padding)
            self.conv7 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                kernel_size=self.kernal_size, padding=self.padding)
        
        self.pool = nn.AdaptiveMaxPool2d((1,1))

        self.fc1 = nn.Linear(in_features=self.inplanes, out_features=self.inplanes) 
        self.fc2 = nn.Linear(in_features=self.inplanes, out_features=2)

    def forward(self, x):
        if self.L3:
            x = self.conv1(x)
            if self.divnorm: x = self.div1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        
        if self.L5:
            x = self.conv4(x)
            x = self.conv5(x)
        
        if self.L7:
            x = self.conv6(x)
            x = self.conv7(x)
        
        x = self.pool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

