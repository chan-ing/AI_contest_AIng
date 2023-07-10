# 상위 5% 하위 5% 건물영역을 지니고 있는 데이터 자르기.  >> csv 파일 생성 
import random
import numpy as np
import pandas as pd
import csv

def make_remove_exception_csvfile(filename):
    data =(pd.read_csv(filename)) #data type : pandas - dataFrame 

    mask_rle_column = data['mask_rle']

    dictionary = data.set_index('img_id')['img_path'].to_dict()
    dictionary = {k: 0 for k in dictionary.keys()}

    for i in range(len(mask_rle_column)):  # 0~7139 ,  7140번 반복 (train data set 개수)
        data_list = mask_rle_column[i].split()            # 레이블 값 > 리스트 
        buildingarea=sum(int(data_list[j]) for j in range(1, len(data_list), 2))
        percent = round(buildingarea/(1024*1024),3)
        dictionary[data['img_id'][i]] = percent

    tmp = []
    for key, value in dictionary.items(): 
        num = int(key.split("_")[1])  # "TRAIN_" 뒤의 숫자만 추출
        if float(value) < 0.012 or float(value) > 0.137:
            if random.random() < 0.5:  # 50%의 확률로 tmp 리스트에 추가
                tmp.append(num)

    # tmp 리스트 출력 (삭제되는 데이터) 
    print(tmp)
    print(len(tmp)) 

    # 특정 행 삭제
    data = data.drop(tmp) 

    data.to_csv("./train_removeoutlier.csv", index=False)


make_remove_exception_csvfile("./train.csv")
