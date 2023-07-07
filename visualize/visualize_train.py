import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def dataload(filename):
    data =list(pd.read_csv(filename)) #data type : pandas - dataFrame  ==> list 
    
    address= data[1]
    lst = data[2].split()
    return address, lst

# 0과 1로 구성된 리스트 만들기 
def make_mask_image_for_training(lst):
    mask = [0 for _ in range(1024*1024)]
    for i in range(len(lst)):
        if i%2==0:
            k=int(lst[i])
            for j in range(int(lst[i+1])):
                mask[k]=1
                k+=1            
    mask_image = np.array(mask).reshape((1024, 1024))
    return mask_image

## 구현하고 싶은 것 > text형식으로 주어진 1024x1024코드에 대해서 출력해보는 것. 
## >> 이를 통해 제출하는 csv파일로부터 읽어와서 시각화를 할 수 있게 하는 것이 최종 목표. 시각화!!
def visualize(image, lst, test=False):
    fontsize = 16

    mask = make_mask_image_for_training(lst)
        
    f, ax = plt.subplots(1, 2, figsize=(12,12), squeeze=True)       #행 1 열 2,,  크기 : 12x12,  
    f.tight_layout(h_pad=5, w_pad=5)

#image 출력
    ax[0].imshow(image)
    ax[0].set_title("Original image", fontsize = fontsize)
    
#mask 출력 ) 흑, 황 색으로 표현
    cmap = plt.cm.colors.ListedColormap(['black', 'yellow'])

    ax[1].imshow(mask, cmap=cmap)
    ax[1].set_title("Mask", fontsize = fontsize)
        
    #plt.savefig('sample_augmented_image.png', facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)  >> 저장 코드, 
    plt.show()  # >> 출력 코드. 
    


address , lst = dataload("./tmp.csv")

image = cv2.imread(address)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

visualize(image, lst)


#파일을 저장하고 싶으면 쓸 수 있는 아래 코드들

#cv2.imwrite('./image.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#cv2.imwrite('./mask.png',cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

