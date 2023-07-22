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
from efficientnet_pytorch import EfficientNet

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

class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        # self.model = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        self.model = EfficientNet.from_pretrained('efficientnet-b4') if pretrained else EfficientNet.from_name('efficientnet-b4')


    def forward(self, x):
        features = self.model.extract_features(x)
        return features
    
class eff_UNet(nn.Module):
    def __init__(self):
        super(eff_UNet, self).__init__()
        self.backbone = EfficientNetBackbone(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up5 = double_conv(1792,896)
        self.dconv_up4 = double_conv(896, 448)
        self.dconv_up3 = double_conv(448, 256)

        # self.dconv_up4 = double_conv(1280, 512)
        # self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

        self.Drop_out = nn.Dropout2d(0.2) 

    def forward(self, x):
        x = self.backbone(x)
        x = self.dconv_up5(x)
        
        x = self.upsample(x)   #1280,14,14
        x = self.dconv_up4(x)  #512,14,14

        x = self.upsample(x)  #512,28,28 
        x = self.dconv_up3(x) #256,28,28

        x = self.upsample(x) #256,56,56
        x = self.dconv_up2(x) #128,56,56

        x = self.upsample(x) #128,112,112
        x = self.dconv_up1(x) #64,112,112

        x = self.upsample(x) #64,224,224
        out = self.conv_last(x)

        return out

model = eff_UNet()
summary(model, input_size=(3, 224, 224))