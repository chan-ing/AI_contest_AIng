import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class ImageVisualizer:
    def __init__(self, filename, test=False):
        self.filename = filename
        self.test = test
        self.address, self.lst = self.dataload()
        self.image = self.load_image()
        

    def dataload(self):
        if self.test == False:
            data = list(pd.read_csv(self.filename))
            address = data[1]
            lst = data[2].split()
            return address, lst
        else:
            data =list(pd.read_csv(self.filename)) #data type : pandas - dataFrame  ==> list 
            address= "./test_img/"+data[0]+".png"
            lst = data[1].split()
            return address, lst
        
    def load_image(self):
        image = cv2.imread(self.address)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def make_mask_image_for_training(self):
        mask = [0 for _ in range(1024*1024)]
        for i in range(len(self.lst)):
            if i % 2 == 0:
                k = int(self.lst[i])
                for j in range(int(self.lst[i+1])):
                    mask[k] = 1
                    k += 1
        mask_image = np.array(mask).reshape((1024, 1024))
        return mask_image
    
    def make_mask_image_for_test(self):
        mask= [ 0 for _ in range(224*224)]
        
        if len(self.lst) == 1 and int(self.lst[0]) == -1 :
            mask_image = np.array(mask).reshape((224,224))
            return mask_image
        
        for i in range(len(self.lst)):
            if i%2==0:
                k=int(self.lst[i])
                for j in range(int(self.lst[i+1])):
                    mask[k]=1
                    k+=1            
        mask_image = np.array(mask).reshape((224, 224))
        return mask_image

    def visualize(self):
        fontsize = 16
        if self.test == False:
            mask = self.make_mask_image_for_training()
        else:
            mask = self.make_mask_image_for_test()

        f, ax = plt.subplots(1, 2, figsize=(12, 12), squeeze=True)
        f.tight_layout(h_pad=5, w_pad=5)

        ax[0].imshow(self.image)
        ax[0].set_title("Original image", fontsize=fontsize)

        cmap = plt.cm.colors.ListedColormap(['black', 'yellow'])
        ax[1].imshow(mask, cmap=cmap)
        ax[1].set_title("Mask", fontsize=fontsize)

        plt.show()


visualizer1 = ImageVisualizer("./visualize/tmp.csv")
visualizer2 = ImageVisualizer("./visualize/tmp_test.csv", True)
visualizer1.visualize()
visualizer2.visualize()