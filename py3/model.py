#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:38:31 2023

Define the CNN structure.
"""
__author__ = "Manuel"
__date__ = "Mon Dec  4 08:38:31 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
#> Import modules
from torch.nn import Module, Sequential, Conv2d, ReLU, Linear, Sigmoid
from torch.nn import BatchNorm2d, MaxPool2d

#-----------------------------------------------------------------------------|
#> Define a simple CNN
class SimpleCNN(Module):
    # Model tested with
    # Tile size: 128, 256
    # Convolutional layers: 2, 3
    # -> No significant differences in maximum validation accuracy.
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = self.conv_block(1, 16)
        self.layer2 = self.conv_block(16, 32)
        #self.layer3 = self.conv_block(32, 64)
        # 2 conv layers: 28800, 3 conv layers: 12544
        self.layer4 = Linear(28800, 64)
        self.relu = ReLU()
        self.layer5 = Linear(64, 1)
        self.sigmoid = Sigmoid()
    
    def conv_block(self, in_channels, out_channels, kernel = 3, s = 1, p = 0):
        block = Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size = kernel, stride = s, padding = p
                ),
            
            ReLU(),
            BatchNorm2d(out_channels),
            MaxPool2d(kernel_size = 2, stride = 2)
            )
        
        return block
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        # Flatten the tensor before the fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.sigmoid(x)
        
        return x
    
    def outsize_conv(self, input_size, kernel_size, padding, stride):
        return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

    def outsize_pool(self, input_size, kernel_size, stride):
        return int((input_size - kernel_size) / stride) + 1
