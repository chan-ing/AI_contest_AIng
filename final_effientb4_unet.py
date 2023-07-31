import os
import cv2
import pandas as pd
import numpy as np
from loss import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from torchsummary import summary
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = A.Compose(
    [
        #A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

class ImageDataset(Dataset):
    def __init__(self, transform, img_folder_path, mask_folder_path=None, infer=True):
        self.img_folder_path = img_folder_path
        self.img_file_list = [file for file in os.listdir(img_folder_path) if file.endswith('.png')]
        self.transform = transform
        self.mask_folder_path = mask_folder_path
        if mask_folder_path is not None:
            self.mask_file_list = [file for file in os.listdir(mask_folder_path) if file.endswith('.png')]
        self.infer = infer

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, idx):
        img_filename = self.img_file_list[idx]
        img_path = os.path.join(self.img_folder_path, img_filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        if self.mask_folder_path is not None:
            mask_filename = self.mask_file_list[idx]
            mask_path = os.path.join(self.mask_folder_path, mask_filename)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 0, 1, 0)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            return image, mask# 간단한 U-Net 모델 정의


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True)
#     )

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.3),
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

        self.dconv_up5 = double_conv(1792, 896)
        self.dconv_up4 = double_conv(896, 448)
        self.dconv_up3 = double_conv(448, 256)

        # self.dconv_up4 = double_conv(1280, 512)
        # self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)


    def forward(self, x):
        x = self.backbone(x)
        x = self.dconv_up5(x)

        x = self.upsample(x)  # 1280,14,14
        x = self.dconv_up4(x)  # 512,14,14

        x = self.upsample(x)  # 512,28,28
        x = self.dconv_up3(x)  # 256,28,28

        x = self.upsample(x)  # 256,56,56
        x = self.dconv_up2(x)  # 128,56,56

        x = self.upsample(x)  # 128,112,112
        x = self.dconv_up1(x)  # 64,112,112

        x = self.upsample(x)  # 64,224,224
        out = self.conv_last(x)

        return out
#---------------------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
#loss = criterion(outputs, masks.unsqueeze(1))
#-----------------------------------------------------------------

if __name__ == '__main__':
    #torch.multiprocessing.freeze_support()

    img_folder_path = "./patch_train_img"
    mask_folder_path = "./patch_train_mask_img"
    dataset = ImageDataset(transform=transform, img_folder_path=img_folder_path, mask_folder_path=mask_folder_path,
                           infer=False)
    # subset_dataset = torch.utils.data.Subset(dataset, range(1000))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    # model 초기화
    #model = eff_UNet().to(device)
    model = eff_UNet().to(device)
    loaded_weights = torch.load("final_effi_Unet_diceBCE_110.pt", map_location=device)
    model.load_state_dict(loaded_weights)


    # loss function과 optimizer 정의
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0.000001)
    # training loop
    for epoch in range(10):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            #loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')
        scheduler.step()

        torch.save(model.state_dict(), f'final_effi_Unet_diceBCE_{epoch}.pt')
    img_folder_path = "./test_img"
    test_dataset = ImageDataset(transform=transform, img_folder_path=img_folder_path, infer=True)
    # subset_test_dataset = torch.utils.data.Subset(test_dataset, range(10000))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.15).astype(np.uint8)  # Threshold = 0.35

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./submit.csv', index=False)