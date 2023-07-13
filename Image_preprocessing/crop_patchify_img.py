import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import random
from tqdm import tqdm


class CropPatchifyImage:
    def __init__(self, folder_path):
        self.random_seed = 761
        self.folder_path = folder_path
        self.scaler = MinMaxScaler()
        self.patch_size = 224
        self.Size_x = (1024 // self.patch_size) * self.patch_size
        self.Size_y = (1024 // self.patch_size) * self.patch_size
        self.new_folder_path = ''

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def crop_patchify(self):
        cnt = 0
        random.seed(self.random_seed)
        folder_path = self.folder_path
        last_dir_name = os.path.basename(folder_path)
        self.new_folder_path = './patch_' + str(last_dir_name)
        self.createFolder(self.new_folder_path)

        image_files = []
        for path, subdir, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.png'):
                    image_files.append(os.path.join(path, file))

        progress_bar = tqdm(total=len(image_files), desc="Processing", unit="image")
        for image_path in image_files:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            rand_x = random.randint(0, 127)
            rand_y = random.randint(0, 127)
            image = image.crop((rand_x, rand_y, rand_x + self.Size_x, rand_y + self.Size_y))
            image = np.array(image)
            patches_img = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)

            for k in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    patch = patches_img[k, j, 0]

                    patch_filename = f"patch_{cnt:0>4}_{k}_{j}.png"
                    patch_filepath = os.path.join(self.new_folder_path, patch_filename)

                    patch_image = Image.fromarray(patch)
                    # patch = self.scaler.fit_transform(patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
                    patch_image.save(patch_filepath)

            cnt += 1
            progress_bar.update(1)

        progress_bar.close()


# # 클래스 인스턴스 생성 및 이미지 처리 실행
# folder_paths = ['./train_img', './train_mask_img']
# for folder_path in folder_paths:
#     cropping = CropPatchifyImage(folder_path)
#     cropping.crop_patchify()