# encoded data > image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class MaskImageCreator:
    def __init__(self, filename):
        self.filename = filename

    def make_mask_image_for_training(self, lst):
        mask = [0 for _ in range(1024 * 1024)]
        for i in range(len(lst)):
            if i % 2 == 0:
                k = int(lst[i]) - 1
                for j in range(int(lst[i + 1])):
                    mask[k] = 1
                    k += 1
        mask_image = np.array(mask).reshape((1024, 1024))
        return mask_image

    def create_mask_image(self):
        data = pd.read_csv(self.filename)
        mask_rle_column = data['mask_rle']
        cmap = plt.cm.colors.ListedColormap(['black', 'yellow'])

        output_folder = "./train_mask_img"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in tqdm(range(len(mask_rle_column)), desc="Processing", unit="image"):
            data_list = mask_rle_column[i].split()
            mask = self.make_mask_image_for_training(data_list)
            path = os.path.join(output_folder, f"/MASK_{i:04}.png")
            plt.imsave(path, mask, cmap=cmap)


# # Usage example
# mask_creator = MaskImageCreator("./train.csv")
# mask_creator.create_mask_image()
