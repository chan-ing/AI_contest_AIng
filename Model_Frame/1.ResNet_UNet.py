import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
import torchvision.models as models

import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResNetBackbone_A(nn.Module):
    def __init__(self):
        super(ResNetBackbone_A, self).__init__()

        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.upsample = nn.Upsample(size=(14, 14), mode='bilinear', align_corners=True)
        
        self.res_down1 = double_conv(2048,1024)
        self.res_down2 = double_conv(1024,512)

    def forward(self, x):
        features = self.features(x)
        features = self.upsample(features)
        features = self.res_down1(features)
        features = self.res_down2(features)

        return features
    
class UNet_A(nn.Module):
    def __init__(self):
        super(UNet_A, self).__init__()
        self.backbone = ResNetBackbone_A()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up1 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up3 = double_conv(128, 64)
        

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.backbone(x)   # 512,14,14
        x = self.upsample(x)  # 512,28,28

        conv1 =self.dconv_up1(x) #256,14,14
        x = self.upsample(conv1)  # 256,56,56

        conv2 = self.dconv_up2(x)  # 128  
        x = self.upsample(conv2)  # 128,112,112

        conv3 = self.dconv_up3(x)  # 64,56,56
        x = self.upsample(conv3) # 64,224,224

        out = self.conv_last(x)

        return out
    
model = UNet_A()
summary(model, input_size=(3, 224, 224))