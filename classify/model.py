import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

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

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.res_down1 = double_conv(2048, 64)

    def forward(self, x):
        features = self.features(x)
        features = self.upsample(features)
        features = self.res_down1(features)
        return features

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.backbone = ResNetBackbone()

        self.dconv_down1 = double_conv(64, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.backbone(x)   # 3 -> 64

        conv1 = self.dconv_down1(x)   # 64 -> 64
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)  # 64 -> 128
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)  # 128 -> 256
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)  # 256 -> 512

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')

    def forward(self, x):
        features = self.model.extract_features(x)
        return features

class eff_UNet(nn.Module):
    def __init__(self):
        super(eff_UNet, self).__init__()
        self.backbone = EfficientNetBackbone(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(1280, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

        self.Drop_out = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.backbone(x)

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
class UNetpp(nn.Module):
    def __init__(self):
        super(UNetpp, self).__init__()
    
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_down = double_conv(3, 32)
        # ~ 0
        self.dconv_down0_0 = double_conv(32, 32)
        self.dconv_down1_0 = double_conv(32, 64)
        self.dconv_down2_0 = double_conv(64, 128)
        self.dconv_down3_0 = double_conv(128, 256)
        self.dconv_down4_0 = double_conv(256, 512)

        # ~ 1
        self.dconv_down0_1 = double_conv(32+64, 32)
        self.dconv_down1_1 = double_conv(64+128, 64)
        self.dconv_down2_1 = double_conv(128+256, 128)
        self.dconv_down3_1 = double_conv(256+512, 256)

        #~ 2
        self.dconv_down0_2 = double_conv(64+64, 32)
        self.dconv_down1_2 = double_conv(128+128, 64)
        self.dconv_down2_2 = double_conv(256+256, 128)

        #~ 3
        self.dconv_down0_3 = double_conv(96+64, 32)
        self.dconv_down1_3 = double_conv(192+128, 64)

        #~ 4
        self.dconv_down0_4 = double_conv(128+64,32)
        
        
        self.output1 = nn.Conv2d(32, 1, 1)
        self.output2 = nn.Conv2d(32, 1, 1)
        self.output3 = nn.Conv2d(32, 1, 1)
        self.output4 = nn.Conv2d(32, 1, 1)

        self.Drop_out = nn.Dropout2d(0.2) 

    def forward(self, x):
        x = self.dconv_down(x)  #32,224,224

        x0_0 = self.dconv_down0_0(x)   #32,224,224
        x = self.maxpool(x0_0)           #32,112,112
        x1_0 = self.dconv_down1_0(x)       #64,112,112
        x = self.upsample(x1_0)
        x = torch.cat([x0_0, self.upsample(x1_0)], dim=1)  #64+32,224,224
        x0_1 = self.dconv_down0_1(x)   #32,224,224

        x = self.maxpool(x1_0)  #64,56,56
        x2_0 = self.dconv_down2_0(x)   #128,56,56
        x = torch.cat([x1_0,self.upsample(x2_0)],dim=1)  #64+128,112,112
        x1_1 = self.dconv_down1_1(x)  #64,112,112
        x = torch.cat([x0_0,x0_1,self.upsample(x1_1)], dim=1) #32+32+64,224,224
        x0_2 = self.dconv_down0_2(x)  #32,224,224
        
        x = self.maxpool(x2_0)
        x3_0 = self.dconv_down3_0(x)
        x = torch.cat([x2_0,self.upsample(x3_0)], dim=1)
        x2_1 = self.dconv_down2_1(x)
        x = torch.cat([x1_0,x1_1,self.upsample(x2_1)], dim=1)
        x1_2 = self.dconv_down1_2(x)
        x = torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], dim=1)
        x0_3 = self.dconv_down0_3(x)

        x = self.maxpool(x3_0)
        x4_0 = self.dconv_down4_0(x)
        x = torch.cat([x3_0,self.upsample(x4_0)], dim=1)
        x3_1 = self.dconv_down3_1(x)
        x = torch.cat([x2_0,x2_1,self.upsample(x3_1)], dim=1)
        x2_2 = self.dconv_down2_2(x)
        x = torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], dim=1)
        x1_3 = self.dconv_down1_3(x)
        x = torch.cat([x0_0, x0_1, x0_2,x0_3 ,self.upsample(x1_3)], dim=1)
        x0_4 = self.dconv_down0_4(x)

        
        output1 = self.output1(x0_1)
        output2 = self.output1(x0_2)
        output3 = self.output1(x0_3)
        output4 = self.output1(x0_4)

        output = (output1 + output2 + output3 + output4)/4

        return output

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(x)
        output3 = self.model3(x)
        ensemble_output = (output1 + output2 + output3) / 3
        return ensemble_output
