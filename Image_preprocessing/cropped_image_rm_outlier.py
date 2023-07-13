# Description : 
# crop된 image가 담긴 Train_img, Mask_img 폴더 중  
# mask_img를 encoded data로 바꾸어 건물이 차지하는 영역을 구해 딕셔너리에 저장 
# 상위 하위 5% 인 파일들을 이상치를 벗어난 값으로 판단하여 randomly하게 train_img 와 mask_img에서 이미지 파일을 제거한다!
# 새로운 폴더등을 생성하지 않고 인자로 전달한 파일내에서 삭제작업을 진행함.
import os
import random
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm


class RemoveException_for_CroppedImage:
    def __init__(self, train_folder_path, mask_folder_path):
        self.train_folder_path = train_folder_path  # path (train_img)
        self.mask_folder_path = mask_folder_path  # path (mask_img)
        self.dict_percent = {}  # to save percentage of building area

    def rle_encode(self, mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)  # remove image file.

    def removeException(self):
        mask_img = os.listdir(self.mask_folder_path)

        # image --> encoded data --> calculate percentage of building area --> dictionary update.
        for i in tqdm(range(len(mask_img)), desc="Processing", unit="image"):
            mask_img_path = os.path.join(self.mask_folder_path, mask_img[i])

            # image --> encoded data;
            img = mpimg.imread(mask_img_path)
            img_array = np.array(img)
            img_array = img_array[:, :, 0]
            mask_rle = self.rle_encode(img_array).split()  # encoded data.

            # dictionary update. (key : image_filename, value : percentage of building area)
            building_area = sum(int(mask_rle[j]) for j in range(1, len(mask_rle), 2))
            percent = round(building_area / (224 * 224), 4)
            self.dict_percent[mask_img[i]] = percent

        # calculate upper and lower bounds
        values = list(self.dict_percent.values())
        top_5_percent = sorted(values)[-int(len(values) * 0.05)]
        # bottom_5_percent = sorted(values)[int(len(values) * 0.05)]

        for mask_file_name, percent in self.dict_percent.items():
            if percent > top_5_percent:
                if random.random() > 0.5:
                    # ○●○●○●○●○●○●○●○●○●○●수정대상○●○●○●○●○●○●○●○●○●○●○●○●○●○● 아래부분
                    # ↓ 제거할 파일 이름에 맞게 끔 수정해야함 . (현재는 삭제 대상인file명이 TEST_00000.png의 형태임.)
                    train_file_name = mask_file_name  # 이후 수정 대상 : 저장된 파일명에 맞게끔
                    train_file_path = os.path.join(self.train_folder_path, train_file_name)
                    mask_file_path = os.path.join(self.mask_folder_path, mask_file_name)
                    self.delete_file(train_file_path)
                    self.delete_file(mask_file_path)
            if percent == 0.0:
                if random.random() < 0.9:
                    train_file_name = mask_file_name.replace("MASK", "TEST")  # 이후 수정 대상 : 저장된 파일명에 맞게끔
                    train_file_path = os.path.join(self.train_folder_path, train_file_name)
                    mask_file_path = os.path.join(self.mask_folder_path, mask_file_name)
                    self.delete_file(train_file_path)
                    self.delete_file(mask_file_path)
                    # If want to see the files being deleted, run the code below.
                    # print(mask_file_path)
                    # print(train_file_path)
        print("현재 이미지 파일 개수:", len(os.listdir(self.mask_folder_path)))

    # 그래프 생성 코드
    def showGraph(self):
        mask_img = os.listdir(self.mask_folder_path)

        # image --> encoded data --> calculate percentage of building area --> dictionary update.
        for i in range(len(mask_img)):
            mask_img_path = os.path.join(self.mask_folder_path, mask_img[i])

            # image --> encoded data;
            img = mpimg.imread(mask_img_path)
            img_array = np.array(img)
            img_array = img_array[:, :, 0]
            mask_rle = self.rle_encode(img_array).split()  # encoded data.

            # dictionary update. (key : image_filename, value : percentage of building area)
            building_area = sum(int(mask_rle[j]) for j in range(1, len(mask_rle), 2))

# train_folder_path = './patch_train_img'
# mask_folder_path = './patch_train_mask_img'
#
# processor = RemoveException_for_CroppedImage(train_folder_path, mask_folder_path)
# processor.removeException()
#=========================================================================================
#다른 python code에서 호출하는 방법.
#from cropped_image_rm_outlier import *

#○●○●○●○●○●○●○●○●○● 수정대상 ○●○●○●○●○●○●○●○●○●○●○●○●      파일 경로에 맞게끔.
#train_folder_path = './tmp/test_img'
#train_mask_folder_path = './tmp/mask_img'


#호출
#processor = RemoveException_for_CroppedImage(train_folder_path, train_mask_folder_path)
#processor.removeException()
