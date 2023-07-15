import os
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
import cv2


class Augmentation:
    def __init__(self, image_input_folder, mask_input_folder, image_output_folder, mask_output_folder):
        self.image_input_folder = image_input_folder
        self.mask_input_folder = mask_input_folder
        self.image_output_folder = image_output_folder
        self.mask_output_folder = mask_output_folder

        self.transform = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=90, p=0.5),
            ]
        )

    def augment_images(self):
        image_files = [f for f in os.listdir(self.image_input_folder) if f.lower().endswith(".png")]
        mask_files = [f for f in os.listdir(self.mask_input_folder) if f.lower().endswith(".png")]

        total_files = len(image_files)
        progress_bar = tqdm(total=total_files * 3, desc="Augmenting Images", unit="image")


        for image_file, mask_file in zip(image_files, mask_files):
            image_name = os.path.basename(image_file)
            mask_name = os.path.basename(mask_file)

            image_path = os.path.join(self.image_input_folder, image_file)
            mask_path = os.path.join(self.mask_input_folder, mask_file)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # Rename existing files
            existing_image_new_name = "augmentation_0_" + image_name
            existing_mask_new_name = "augmentation_0_" + mask_name
            existing_image_output_path = os.path.join(self.image_output_folder, existing_image_new_name)
            existing_mask_output_path = os.path.join(self.mask_output_folder, existing_mask_new_name)
            os.rename(image_path, existing_image_output_path)
            os.rename(mask_path, existing_mask_output_path)

            for i in range(1, 4):

                transformed = self.transform(image=image, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

                # Save augmented images with "augmentation_{i}_"
                augmented_image_new_name = f"augmentation_{i}_" + image_name
                augmented_mask_new_name = f"augmentation_{i}_" + mask_name

                augmented_image_output_path = os.path.join(self.image_output_folder, augmented_image_new_name)
                augmented_mask_output_path = os.path.join(self.mask_output_folder, augmented_mask_new_name)

                plt.imsave(augmented_image_output_path, transformed_image, format='png')
                plt.imsave(augmented_mask_output_path, transformed_mask, format='png')


                progress_bar.update(1)

        progress_bar.close()
# # 경로 설정
# image_folder = "./patch_train_img"
# mask_folder = "./patch_train_mask_img"
#
# # ImageAugmenter 인스턴스 생성
# augmenter = Augmentation(image_folder, mask_folder, image_folder, mask_folder)
#
# # 이미지 증강 수행
# augmenter.augment_images()