import os
import cv2
import numpy as np
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths
        self.png_file_count = 0

    def count_png_files(self, folder_path):
        png_file_count = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(".png"):
                    png_file_count += 1
        return png_file_count

    def process_images(self):
        print("Checking Cropped & Patchfied images")
        for folder_path in self.folder_paths:
            png_file_count = self.count_png_files(folder_path)
            print("검사 directory :", folder_path)

            progress_bar = tqdm(total=png_file_count, desc="Processing", unit="file")
            progress_bar.clear()
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith(".png"):
                        image = cv2.imread(file_path)
                        image_array = np.array(image)

                        if image_array.shape != (224, 224, 3):
                            print(file, "error 발생!!!\n shape :", image_array.shape)
                            continue

                        progress_bar.update(1)

            progress_bar.close()
            print("이미지 파일 개수 :", png_file_count)

            need_file_cnt = 7140                               # 기존 train image의 개수
            if png_file_count == need_file_cnt * 16:
                print("----------모든 이미지 파일 성공-----------\n")
            else:
                print("!!!!!!!이미지 개수가 모자릅니다!!!!!!!!\n")

# # 클래스 인스턴스 생성 및 이미지 처리 실행
# folder_paths = ["./patch_train_mask_img", "./patch_train_img"]  # 처리할 폴더 경로들을 리스트로 지정합니다.
# image_processor = ImageProcessor(folder_paths)
# image_processor.process_images()
