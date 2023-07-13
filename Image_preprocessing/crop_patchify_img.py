#참조
#https://github.com/bnsreenu/python_for_microscopists/blob/master/228_semantic_segmentation_of_aerial_imagery_using_unet/228_training_aerial_imagery.py

import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image

import random


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

#createFolder('/Users/aaron/Desktop/test')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
random.seed(761)

def crop_image(filename):       #filename --> train_img, train_mask_img
    scaler = MinMaxScaler()
    cnt = 0
    patch_size = 224                            # patch_size 설정 (224X224)
    root_path = './data/' + str(filename)       # input image 경로 지정
    Size_x = (1024 // patch_size) * patch_size  # Nearest size divisible by our patch size
    Size_y = (1024 // patch_size) * patch_size  # Nearest size divisible by our patch size
    new_folder_path = './normalize_crop_'+filename
    createFolder(new_folder_path)
    #해당 path 이하의 모든 파일에서 png파일 접근
    for path, subdir, files in os.walk(root_path):
        images = os.listdir(path)
        for i, image_name in enumerate(images):
            if image_name.endswith('.png'):
                #print(path + "/" + image_name)
                image = cv2.imread(path + "/" + image_name)  # Read each image as BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

                rand_x = random.randint(0, 127)
                rand_y = random.randint(0, 127)
                image = image.crop((rand_x, rand_y, rand_x+Size_x, rand_y+Size_y))  # Crop from top left corner
                image = np.array(image)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

                for k in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        patch = patches_img[k, j, 0]

                        #patch = patch[0]
                        patch_filename = "patch_{:0>4}_{}_{}.png".format(cnt, k, j)  # 저장할 패치 파일명 설정
                        patch_filepath = os.path.join(new_folder_path, patch_filename)
                        # 이미지 저장 코드 (PNG 형식으로 저장)

                        patch_image = Image.fromarray(patch)
                        patch = scaler.fit_transform(patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
                        print(patch)
                        patch_image.save(patch_filepath)
                cnt += 1
                #image.save('./crop/tr'+str(i).zfill(4)+'.png', 'png')
crop_image("train_img")

"""    image = Image.fromarray(image)
    image = image.crop((0, 0, Size_x, Size_y))  # Crop from top left corner
    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    image = np.array(image)
"""

