import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from preprocess import *
from model import UNet, eff_UNet, EnsembleModel, UNetpp
from loss import FocalLoss, LovaszLoss

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose(
        [
            #A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    img_folder_path = "./patch_train_img"
    mask_folder_path = "./patch_mask_img"

    dataset = ImageDataset(transform=transform, img_folder_path=img_folder_path, mask_folder_path=mask_folder_path, inder=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)
    
    model1 = UNet().to(device)
    model2 = eff_UNet().to(device)
    model3 = UNetpp().to(device)

    ensemble_model = UNet().to(device)
    eff_unet_model = eff_UNet().to(device)

    criterion = FocalLoss(alpha=0.5, gamma=2)
    #criterion = LovaszLoss()
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.001)

    for epoch in range(5):
        ensemble_model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = ensemble_model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            #loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

    img_folder_path = "./test_img"
    
    test_dataset = ImageDataset(transform=transform, img_folder_path=img_folder_path)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

    with torch.no_grad():
        EnsembleModel.eval()

        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            outputs = ensemble_model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./submit.csv', index=False)
