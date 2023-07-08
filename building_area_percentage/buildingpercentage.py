import numpy as np
import pandas as pd
import csv

data =(pd.read_csv("./train.csv")) #data type : pandas - dataFrame  ==> list 
#print(data)

mask_rle_column = data['mask_rle']
#print(mask_rle_column[0].split()) ## list  i번째 행에 대해 접근하는 
dictionary = data.set_index('img_id')['img_path'].to_dict()
dictionary = {k: 0 for k in dictionary.keys()}

for i in range(len(dictionary)):  # 0~7139 ,  7140번 반복 (train data set 개수)
    data_list = mask_rle_column[i].split()            # 레이블 값 > 리스트 
    buildingarea=sum(int(data_list[j]) for j in range(1, len(data_list), 2))
    percent = round(buildingarea/(1024*1024),3)
    dictionary[data['img_id'][i]] = percent
    
with open("percentage.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["img_id", "building_percentage"])  # 헤더 쓰기

    for img_id, building_percentage in dictionary.items():
        writer.writerow([img_id, building_percentage])

print(123)