
import os
import random
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class MakeValidationSet:
    def __init__(self, train_img_folder, mask_img_folder, num_samples=80000):
        self.random_seed = 761
        self.train_img_folder = train_img_folder
        self.mask_img_folder = mask_img_folder
        self.validation_img_folder = "./Valid_" + train_img_folder.split('/')[1]
        self.validation_mask_img_folder = "./Valid_" + mask_img_folder.split('/')[1]
        self.num_samples = num_samples
        

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def make_validation(self):
        random.seed(self.random_seed)

        self.create_folder(self.validation_img_folder)
        self.create_folder(self.validation_mask_img_folder)

        image_files = [file for file in os.listdir(self.train_img_folder) if file.endswith('.png')]
        random.shuffle(image_files)

        progress_bar = tqdm(total=self.num_samples, desc="Moving Data", unit="image")

        for i in range(self.num_samples):
            # patch_train_img와 patch_mask_img에 저장된 파일은 각각 이름이 같음.
            image_filename = image_files[i]
            mask_filename = image_filename  
            image_src = os.path.join(self.train_img_folder, image_filename)
            mask_src = os.path.join(self.mask_img_folder, mask_filename)

            image_dst = os.path.join(self.validation_img_folder, "Valid_" + image_filename)
            mask_dst = os.path.join(self.validation_mask_img_folder, "Valid_" + mask_filename)
            
            shutil.move(image_src, image_dst)
            shutil.move(mask_src, mask_dst)

            progress_bar.update(1)

        progress_bar.close()
    
    def make_csv(self):
        file_names = os.listdir(self.validation_mask_img_folder)
        df_list = []
        
        progress_bar = tqdm(total=len(file_names), desc="Make CSV File", unit="image")

        for file_name in file_names:
            if file_name.endswith('.png'):
                image_path = os.path.join(self.validation_mask_img_folder, file_name)
                
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                mask_rle = rle_encode(img)

                df_list.append({'img_id': file_name.split('.')[0], 'mask_rle': mask_rle})

                progress_bar.update(1)

        progress_bar.close()

        df = pd.DataFrame(df_list)
        csv_file_path = "./truth.csv"
        df.to_csv(csv_file_path, index=False)
        
# Usage example:
'''
train_img_folder = './patch_train_img'
mask_img_folder = './patch_train_mask_img'
Validset = MakeValidationSet(train_img_folder, mask_img_folder)
Validset.make_validation()
Validset.make_csv()
'''