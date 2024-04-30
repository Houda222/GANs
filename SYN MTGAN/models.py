
import os
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
import sys


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)



class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        
        # Encoder
        in_features = 16
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(
                            in_channels=in_features,
                            out_channels=out_features,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Decoder
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(
                                    in_features,
                                    out_features,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    



class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
    
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))
        
        # Classification layers
        # Fake:Real branch
        self.conv61 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        # Classification branch
        self.conv62 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        
        x = self.conv4(x)
        x = self.conv5(x)

        x1 = self.conv61(x1)
        x1 = F.avg_pool2d(x1, x1.size()[2:]).view(x1.size()[0], -1)

        x2 = self.conv62(x)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)

        return x1, x2
