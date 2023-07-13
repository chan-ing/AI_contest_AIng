import os
import glob
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm

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
        image_files = glob.glob(os.path.join(self.image_input_folder, "*.png")) #glob.glob 함수를 이용해 png인 모든 파일 경로를 찾아서 image_files리스트에 저장
        for image_file in tqdm(image_files):   #tqdm: 반복문 진행 상황을 표시하는 라이브러리
            #이미지 파일 경로 설정
            image_name = os.path.basename(image_file)  #image_file의 경로에서 파일명만 추출
            mask_name = "MASK_" + image_name[6:]       #image_name 뒤 숫자 동일하도록 설정 (ex train_'0000' - mask_'0000')
            mask_file = os.path.join(self.mask_input_folder, mask_name) #mask_file 경로 설정
            
            image = plt.imread(image_file) #이미지 읽어오기
            mask = plt.imread(mask_file)   #마스크 읽어오기
            
            for i in range(5): #증강시키고 싶은 수
                transformed = self.transform(image=image, mask=mask) #augmentation 수행, transform 함수 : 이미지와 마스크를 입력받아 결과를 딕셔너리 형태로 반환
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                
                image_output_name = "augmentation_" + image_name[:-4] + f"_{i}.png" #지정할 이미지와 파일 이름,경로 설정
                mask_output_name = "augmentation_" + mask_name[:-4] + f"_{i}.png"
                image_output_path = os.path.join(self.image_output_folder, image_output_name) #최종파일 경로 설정
                mask_output_path = os.path.join(self.mask_output_folder, mask_output_name)

                # os.makedirs(self.image_output_folder, exist_ok=True) #폴더가 존재하지 않을때 폴더 생성
                # os.makedirs(self.mask_output_folder, exist_ok=True)
                
                plt.imsave(image_output_path, transformed_image, format='png')  #png파일로 저장
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