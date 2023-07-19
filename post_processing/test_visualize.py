import os
import pandas as pd
from tqdm import tqdm
from PIL import Image


class TestVisualize:
    def __init__(self, filename):
        self.filename = filename

    def make_mask_image_for_test(self, lst):
        if lst[0] == '-1':
            mask = [0] * (224 * 224)
        else:
            mask = [0] * (224 * 224)
            for i in range(len(lst)):
                if i % 2 == 0:
                    k = int(lst[i]) - 1
                    for j in range(int(lst[i+1])):
                        mask[k] = 1
                        k += 1
        mask_image = Image.new('P', (224, 224))
        mask_image.putdata(mask)
        mask_image.putpalette([0, 0, 0, 255, 255, 255])
        return mask_image

    def create_mask_image(self):
        data = pd.read_csv(self.filename)
        mask_rle_column = data['mask_rle']

        output_folder = "./test_mask_img/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in tqdm(range(len(mask_rle_column)), desc="Processing", unit="image"):
            data_list = mask_rle_column[i].split()
            mask_image = self.make_mask_image_for_test(data_list)
            if mask_image is not None:
                path = os.path.join(output_folder, f"MASK_{i:05}.png")
                mask_image.save(path)


# # Usage example
# mask_creator = TestVisualize("./submit.csv")
# mask_creator.create_mask_image()
