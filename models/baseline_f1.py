import torch
import torch.nn as nn
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

        if self.L3:
            self.L5 = False
            self.L7 = False
        elif self.L5:
            self.L3 = True
            self.L7 = False
        elif self.L7:
            self.L3 = True
            self.L5 = True

        self.L3_convs = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding),
                    DivNormExcInh(self.inplanes) if self.divnorm else nn.MaxPool2d(1), 
                    nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding),
                    nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding)
                    )
        self.L5_convs_extra = nn.Sequential(
                    nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding),
                    nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding)
                    )
        self.L7_convs_extra = nn.Sequential(
                    nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding),
                    nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                    kernel_size=self.kernal_size, padding=self.padding)
                    )
        
        self.pool = nn.AdaptiveMaxPool2d((1,1))

        self.fc1 = nn.Linear(in_features=self.inplanes, out_features=self.inplanes) 
        self.fc2 = nn.Linear(in_features=self.inplanes, out_features=2)

    def forward(self, x):   
        if self.L3: 
            x = self.L3_convs(x)
        if self.L5: 
            x = self.L5_convs_extra(x)
        if self.L7: 
            x = self.L7_convs_extra(x)

        x = self.pool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

