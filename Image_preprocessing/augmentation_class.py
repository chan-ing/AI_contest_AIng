import os
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
import cv2

class Augmentation:
    def __init__(self, image_input_folder, mask_input_folder, image_output_folder, mask_output_folder):
        self.image_input_folder = image_input_folder #기존 이미지 불러올 경로
        self.mask_input_folder = mask_input_folder   #기존 마스크 불러올 경로 설정
        self.image_output_folder = image_output_folder  #증강시킨 이미지 저장시킬 경로
        self.mask_output_folder = mask_output_folder    #증강시킨 마스크 저장시킬 경로
        
        self.transform = A.Compose( # augmentation 설정
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=90, p=0.5),
                #A.RandomBrightnessContrast(p=1), #임의로 명암 조절
            ]
        )

    def augment_images(self):
        image_files = [f for f in os.listdir(self.image_input_folder) if f.lower().endswith(".png")]
        mask_files = [f for f in os.listdir(self.mask_input_folder) if f.lower().endswith(".png")]
        for image_file, mask_file in tqdm(zip(image_files, mask_files)):
            image_name = os.path.basename(image_file)
            mask_name = os.path.basename(mask_file)

            image_path = os.path.join(self.image_input_folder, image_file)
            mask_path = os.path.join(self.mask_input_folder, mask_file)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

            for i in range(5):
                transformed = self.transform(image=image, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

                image_output_name = f"augmentation_{i}" + image_name + ".png"
                mask_output_name = f"augmentation_{i}" + mask_name + ".png"
                image_output_path = os.path.join(self.image_output_folder, image_output_name)
                mask_output_path = os.path.join(self.mask_output_folder, mask_output_name)

                plt.imsave(image_output_path, transformed_image, format='png')
                plt.imsave(mask_output_path, transformed_mask, format='png')

# # 경로 설정
# image_folder = "./patch_train_img"
# mask_folder = "./patch_train_mask_img"
#
# # ImageAugmenter 인스턴스 생성
# augmenter = Augmentation(image_folder, mask_folder, image_folder, mask_folder)
#
# # 이미지 증강 수행
# augmenter.augment_images()