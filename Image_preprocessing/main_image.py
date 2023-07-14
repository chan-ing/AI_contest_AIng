from make_mask_img_file import *
from crop_patchify_img import *
from image_checker import *
from cropped_image_rm_outlier import *
from augmentation_class import *

#실행전 현재 디렉토리안에 test.csv, train.csv, test_img 폴더, train_img 폴더가 있어야 합니다.

#결과 : train_mask_img 폴더 생성 후, mask image 저장됨.
mask_creator = MaskImageCreator("./train.csv")
mask_creator.create_mask_image()

#cropping (896*896*3) --> pathchify (224*224*3)
#결과 : 이미지 Crop & patchify 하여 224*224*3 shape의 이미지들이
#"./patch_train_mask_img", "./patch_train_img" 2개의 디렉터리로 각각 저장됨
folder_paths = ['./train_img', './train_mask_img']
for folder_path in folder_paths:
    cropping = CropPatchifyImage(folder_path)
    cropping.crop_patchify()

#위에서 patchify한 이미지들이 올바른지 검사
new_folder_paths = ["./patch_train_mask_img", "./patch_train_img"]
image_processor = ImageProcessor(new_folder_paths)
image_processor.process_images()

#image data outlier 제거하기
#결과 : 이하 경로 안의 이미지들이 일부 제거 됩니다.
train_folder_path = './patch_train_img'
mask_folder_path = './patch_train_mask_img'
processor = RemoveException_for_CroppedImage(train_folder_path, mask_folder_path)
processor.removeException()

#augmentation
#경로 설정
image_folder = "./patch_train_img"
mask_folder = "./patch_train_mask_img"

# ImageAugmenter 인스턴스 생성
augmenter = Augmentation(image_folder, mask_folder, image_folder, mask_folder)

# 이미지 증강 수행
augmenter.augment_images()