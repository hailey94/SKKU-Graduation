import model as modellib
import pandas as pd
#import cv2
import os
import numpy as np
from tqdm import tqdm
from bowl_dataset import BowlDataset
from utils import rle_encode, rle_decode, rle_to_string
import skimage.io
import functions as f
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
from bowl_config import BowlConfig
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation
class InferenceConfig(BowlConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

def refineMasks(mask):
    # import cv2
    # kernel = np.ones((2, 2), np.uint8)
    # masks_post = cv2.dilate(mask, kernel, iterations=2)
    # return masks_post
    return binary_dilation(mask, disk(1))

def eval_mask(pred_masks, gt_masks):
    y_pred = pred_masks.transpose(2, 0, 1)
    y_pred = y_pred.astype(np.uint8)
    gt_masks = gt_masks.transpose(2, 0, 1)
    print(gt_masks.max())
    print(y_pred.max())
    num_masks, height, width = gt_masks.shape

    y_true = np.zeros((num_masks, height, width), np.uint16)
    y_true[:, :, :] = gt_masks[:, :, :]  # Change ground truth mask to zeros and ones

    num_true = len(y_true)
    num_pred = len(y_pred)
    print("Number of true objects:", num_true)
    print("Number of predicted objects:", num_pred)

    # Compute iou score for each prediction
    iou = []
    for pr in range(num_pred):
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for tr in range(num_true):
            olap = y_pred[pr] * y_true[tr]  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(y_pred[pr], y_true[tr]))  # Union formed with sum of maxima
        iou.append(bol / bun)

    # Loop over IoU thresholds
    p = 0
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        matches = iou > t
        tp = np.count_nonzero(matches)  # True positives
        fp = num_pred - tp  # False positives
        fn = num_true - tp  # False negatives
        p += tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, tp / (tp + fp + fn)))

    print("AP\t-\t-\t-\t{:1.3f}".format(p / 10))

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = ['./logs/bowl20180307T2320/mask_rcnn_bowl_0100.h5']
dataset_eval_list=['stage_1_c2_validation']
# Load trained weights (fill in path to trained weights here)
sample_submission = pd.read_csv('split_data_val.csv')
for idx, model_cluster_path in enumerate(model_path):
    assert model_cluster_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_cluster_path)
    model.load_weights(model_cluster_path, by_name=True)

    dataset_val = BowlDataset()
    dataset_val.load_bowl(os.path.join('./dataset', dataset_eval_list[idx]))
    dataset_val.prepare()

    for image_id in sample_submission[sample_submission.hsv_cluster == idx].image_id:
        image_path = os.path.join('./dataset',dataset_eval_list[idx], image_id, 'images', image_id + '.png')

        original_image = skimage.io.imread(image_path)
        if original_image.shape[2] != 3:
            original_image = original_image[:, :, :3]
        results = model.detect([original_image], verbose=0)
        r = results[0]

        pred_masks = r['masks']

        for i in range(pred_masks.shape[2] - 1):
            pred_masks[:, :, i] = refineMasks(pred_masks[:, :, i])
        #
        sum_predicts = np.sum(pred_masks, axis=2)
        rows, cols = np.where(sum_predicts >= 2)

        for i in zip(rows, cols):
            instance_indicies = np.where(np.any(pred_masks[i[0], i[1], :]))[0]
            highest = instance_indicies[0]
            pred_masks[i[0], i[1], :] = pred_masks[i[0], i[1], :] * 0
            pred_masks[i[0], i[1], highest] = 1


        #Gt_masks
        gt_masks, class_id = dataset_val.load_mask_by_name(os.path.join('./dataset', dataset_eval_list[0],image_id))
        eval_mask(pred_masks, gt_masks)
