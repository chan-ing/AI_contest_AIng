import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kornia.augmentation as K
import kornia.augmentation as K

def build_augmentation(self):
        augmentation = K.Compose([
            K.RandomPerspective(distortion_scale=0.5, p=0.5),
            K.RandomAffine(degrees=(-45, 45), scale=(0.8, 1.2), p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=(-45, 45), p=0.5),
            K.RandomCrop(width=256, height=256),
            K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.5),
            K.RandomResizedCrop(width=512, height=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
            K.RandomCrop(width=512, height=512),
            K.Normalize(),
            ToTensorV2(),
        ])

        return augmentation