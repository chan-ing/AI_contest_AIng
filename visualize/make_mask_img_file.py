# encoded data > image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def make_mask_image_for_training(lst):
    mask = [0 for _ in range(1024*1024)]
    for i in range(len(lst)):
        if i%2==0:
            k=int(lst[i])-1
            for j in range(int(lst[i+1])):
                mask[k]=1
                k+=1            
    mask_image = np.array(mask).reshape((1024, 1024))
    return mask_image

def create_mask_image(filename):
    
    data = pd.read_csv(filename)

    mask_rle_column = data['mask_rle']


    cmap = plt.cm.colors.ListedColormap(['black', 'yellow'])


    ## mask_img라는 directory만들고 실행시 마스크 이미지 저장. 
    for i in range(len(mask_rle_column)):
        data_list = mask_rle_column[i].split()
        mask = make_mask_image_for_training(data_list)
        path = "./mask_img/"+"MASK_"+str(i).zfill(4)+".png"
        plt.imsave(path, mask, cmap=cmap)
        if i%100==0:
            print(i)
        
create_mask_image("./train.csv")
    

