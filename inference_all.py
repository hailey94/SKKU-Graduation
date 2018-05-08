import model as modellib
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
# from inference_config import inference_config
from bowl_dataset import BowlDataset
from utils import rle_encode, rle_decode, rle_to_string
import skimage.io

import functions as f

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
from bowl_config import BowlConfig


class InferenceConfig(BowlConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
#-------------------New FPN o.482---------------
model_path = './new_fpn/mask_rcnn_bowl_0099.h5'  # './logs/bowl20180305T0822/mask_rcnn_bowl_0057.h5'#model.find_last()[1]#'./logs/bowl20180306T1818/mask_rcnn_bowl_0090.h5'#model.find_last()[1] #'./logs/bowl20180305T0822/mask_rcnn_bowl_0039.h5'#
#-----------------------------------------------
#-------------------Original FPN o.48---------------
#model_path = './ori_fpn/mask_rcnn_bowl_0099.h5'

assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_test = BowlDataset()
dataset_test.load_bowl('stage1_val')
dataset_test.prepare()

output = []
sample_submission = pd.read_csv('sub_validation.csv')

ImageId = []
EncodedPixels = []
Scores_all = []
for image_id in tqdm(sample_submission.ImageId):
    image_path = os.path.join('stage1_val', image_id, 'images', image_id + '.png')

    original_image = cv2.imread(image_path)
    if original_image.shape[2] != 3:
        original_image = original_image[:, :, :3]
    results = model.detect([original_image], verbose=0)
    r = results[0]

    masks = r['masks']
    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch

f.write2csv('sub_resnet101_Adam_new_FPN_val.csv', ImageId, EncodedPixels)
