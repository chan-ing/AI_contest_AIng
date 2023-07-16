import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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
        
transform = A.Compose(
    [
        # A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)