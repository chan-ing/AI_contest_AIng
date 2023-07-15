# 마스크 데이터가 이미지일때 데이터를 로드하는 클래스.
# 해당 파일을 import 하는 파일에서는 transform이 다르게 정의되어 있지 않은지 한번씩 확인 부탁드립니다.
# dataloader 하는 방법 등은 아래에 작성해두었습니다.
# 이후 학습 및 테스트 코드는 동일합니다.

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2()
        ]
    )
class ImageDataset(Dataset):
    def __init__(self, transform, img_folder_path, mask_folder_path = None, infer=True):
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

            return image, mask
       
       
#=======================================================# 
#from ImageDataset import *

#적용방법 train data --> (인자 : 변환함수, train_img저장된 폴더, mask_img저장된 폴더, infer = False ) 4개 모두 입력해야함.

#경로지정 
'''
img_folder_path="./patch_train_img" 
mask_folder_path="./patch_mask_img"

dataset = ImageDataset(transform=transform, img_folder_path=img_folder_path, mask_folder_path= mask_folder_path, infer=False)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)    #batch_size는 유동적 
'''
#테스트 데이터 적용방법 (인자 : 변환함수, test_img저장된 폴더) 2개 모두 입력해야함. 

#경로지정
'''
img_folder_path = "./test_img"

test_dataset = ImageDataset(transform=transform, img_folder_path=img_folder_path)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)  #batch_size는 유동적

'''