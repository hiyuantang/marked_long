import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from models.dale_rnn_layer import *
from models.utils import get_gabor_conv

#Dale-RNN
class DaleRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernal_size = 5
        self.padding = 2
        self.inplanes = 32

        self.L3_convs = nn.Sequential(
                        #get_gabor_conv(in_channels=self.in_channels, out_channels=self.inplanes, f_size=11, stride=2), 
                        nn.Conv2d(in_channels=3, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding),
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU(), 
                        #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        DaleRNNLayer(self.inplanes, self.inplanes, 3, 5, 3),           
                        nn.Conv2d(in_channels=self.inplanes, out_channels=self.inplanes, 
                                        kernel_size=self.kernal_size, padding=self.padding), 
                        nn.BatchNorm2d(self.inplanes), 
                        nn.ReLU(), 
                        nn.Conv2d(in_channels=self.inplanes, out_channels=2, 
                                        kernel_size=self.kernal_size, padding=self.padding), 
                        #nn.BatchNorm2d(self.inplanes), 
                        #nn.ReLU()
                        )

    def forward(self, x):  
        x = self.L3_convs(x)

        return x

def dalernn(num_outputs, depth, width, dataset):
    return DaleRNN()