from numpy import identity
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

        if self.L3:
            self.L3_convs = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding),
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU(), 
                        DivNormExcInh(self.inplanes) if self.divnorm else nn.Identity(), 
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding),
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU(), 
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding), 
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU()
                        )
        if self.L5:
            self.L5_convs_extra = nn.Sequential(
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding),
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU(), 
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding), 
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU()
                        )
        if self.L7:
            self.L7_convs_extra = nn.Sequential(
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding),
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU(), 
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding), 
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU()
                        )
        
        self.pool = nn.AdaptiveMaxPool2d((1,1))

        self.fc1 = nn.Linear(in_features=self.inplanes, out_features=2)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, DivNormExcInh) or 'divnorm' in n:
                print("Not initializing Divnorm layer weights")
                continue
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

        return x

