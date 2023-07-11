
import os
import cv2
import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image

# 입력 디렉토리 및 출력 디렉토리 설정
image_input_folder = "./train_img"
mask_input_folder = "./mask_img"
image_output_folder = "./image_augmentation_img"
mask_output_folder = "./mask_augmentation_img"
#같은 폴더에 저장하고 싶으면 output폴더 경로 바꾸면됨

# augmentation 설정
transform = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=1.0, rotate_limit=45, p=1.0),  #scale_limit 1.0으로 설정하여 크기변화 X
        A.Rotate(limit=45, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0), #명암 대비 무작위 변환
        #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0), #이미지 색조, 채도, 명도 조절
    ]
)

# 입력 디렉토리의 모든 이미지 파일에 대해 augmentation 적용 및 저장
image_files = glob.glob(os.path.join(image_input_folder, "*.png"))
for image_file in tqdm(image_files):
    # 이미지 파일 경로 설정
    image_name = os.path.basename(image_file)
    mask_name = "MASK_" + image_name[6:]
    mask_file = os.path.join(mask_input_folder, mask_name)
    
    # 이미지 로드
    image = plt.imread(image_file)
    mask = plt.imread(mask_file)
    
    # augmentation 적용 및 저장
    for i in range(5):
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        # 저장할 파일명 설정
        image_output_name = "augmentation_" + image_name[:-4] + f"_{i}.png"
        mask_output_name = "augmentation_" + mask_name[:-4] + f"_{i}.png"
        image_output_path = os.path.join(image_output_folder, image_output_name)
        mask_output_path = os.path.join(mask_output_folder, mask_output_name)
        
        # 이미지 저장
        plt.imsave(image_output_path, transformed_image, format='png')
        plt.imsave(mask_output_path, transformed_mask, format='png')

#출처 https://albumentations.ai/docs/getting_started/mask_augmentation/