import os
from tqdm import tqdm
from test_visualize import TestVisualize
from closing_opening import MorphOpenClose
from Ramer_Douglas_algo import PolygonApproximator
import pandas as pd

import numpy as np


#!!!!! 현재 디렉토리에 submit.csv 파일이 있어야됩니다. !!!!!
def gray_to_binary_mask(gray_image):
    # 이진 마스크로 변환
    binary_mask = (gray_image > 0.5).astype(np.uint8)
    return binary_mask

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



if __name__ == "__main__":
    filename = "submit.csv"  # csv 파일 이름을 여기에 넣으세요

    # TestVisualize 객체 생성 및 mask 이미지 생성
    print("TestVisualize 객체 생성 및 mask 이미지 생성 중...")
    test_visualizer = TestVisualize(filename)
    test_visualizer.create_mask_image()
    print("TestVisualize 객체 생성 및 mask 이미지 생성 완료")

    # 입력 디렉토리와 출력 디렉토리 설정
    input_directory = "./test_mask_img/"
    output_directory = "./post_test_mask_img/"

    # 만약 출력 디렉토리가 없다면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # MorphOpenClose 객체 생성
    morph_processor = MorphOpenClose(kernel_size=5, opening_iterations=2, closing_iterations=0)

    # PolygonApproximator 객체 생성
    polygon_approximator = PolygonApproximator()

    result = []
    # 입력 디렉토리 내의 모든 PNG 파일에 대해 이미지 처리 작업 수행
    for filename in tqdm(os.listdir(input_directory), desc="진행 중", unit="이미지"):
        if filename.endswith(".png"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"post_{filename}")

            # 이미지 처리 수행 (opening과 closing)
            closed_image = morph_processor.opening_closing(input_file)

            # 다각형 근사화 적용 (epsilon 값은 0.01로 사용)
            output_image = polygon_approximator.approximate_polygon(closed_image, epsilon=0.1)

            # 다각형 근사화 결과 이미지 저장
            output_polygon_file = os.path.join(output_directory, f"post_polygon_{filename}")

            #gray_image = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

            # 그레이스케일 이미지를 이진 마스크로 변환
            binary_mask = gray_to_binary_mask(output_image)

            # 이진 마스크를 RLE 인코딩
            encoded_mask = rle_encode(binary_mask)
            #cv2.imwrite(output_polygon_file, output_image)

            if encoded_mask == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(encoded_mask)
            #print(encoded_mask)
    print(len(result))
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./real_submit.csv', index=False)