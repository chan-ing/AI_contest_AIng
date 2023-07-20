import numpy as np
import pandas as pd
from typing import List, Union
from joblib import Parallel, delayed

def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    if mask_rle == -1:
        return np.zeros(shape)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)


def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)) -> List[float]:
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]  # img_id가 같은 것만 취급 
    prediction_df.index = range(prediction_df.shape[0])

    pred_mask_rle = prediction_df.iloc[:, 1]  #인코딩된 data. 
    gt_mask_rle = ground_truth_df.iloc[:, 1]


    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)  # 인코딩 된 데이터를 > 디코딩 시킴. 
        gt_mask = rle_decode(gt_rle, img_shape)


        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None


    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )


    dice_scores = [score for score in dice_scores if score is not None]  # Exclude None values


    return np.mean(dice_scores)



## 사용방법
#ground_truth_df : 정답이 되는 마스킹 csv file
ground_truth_df = pd.read_csv("./truth.csv")
#prediction_df : 모델 학습 이후 validation 데이터의 예측 csv file  (기존으로 따지면 submit.csv file이 될 것임)
#valid data set을 평가 이후 저장된 csv file을 입력하면 됨.
prediction_df = pd.read_csv("./prediction.csv")
print(calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)))