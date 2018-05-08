import model as modellib
import pandas as pd
#import cv2
import os
import numpy as np
from tqdm import tqdm
#from inference_config import inference_config
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
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]#'./logs/bowl20180306T1818/mask_rcnn_bowl_0090.h5'#model.find_last()[1] #'./logs/bowl20180305T0822/mask_rcnn_bowl_0039.h5'#

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_test = BowlDataset()
dataset_test.load_bowl('stage1_test')
dataset_test.prepare()

output = []
sample_submission = pd.read_csv('stage1_sample_submission.csv')

ImageId = []
EncodedPixels = []
Scores_all=[]
for image_id in tqdm(sample_submission.ImageId):
    image_path = os.path.join('stage1_test', image_id, 'images', image_id + '.png')

    original_image = skimage.io.imread(image_path)
    if original_image.shape[2] != 3:
        original_image = original_image[:, :, :3]
    results = model.detect([original_image], verbose=0)
    r = results[0]

    masks = r['masks']
    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch

f.write2csv('submission_v13_coco_newpost_from_44.csv', ImageId, EncodedPixels)


# for image_id in tqdm(sample_submission.ImageId):
#     image_path = os.path.join('stage1_train', image_id, 'images', image_id + '.png')
#
#     original_image = skimage.io.imread(image_path)
#     if original_image.shape[2] != 3:
#         original_image = original_image[:, :, :3]
#     results = model.detect([original_image], verbose=0)
#     r = results[0]
#
#     masks = r['masks']
#     ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
#     ImageId += ImageId_batch
#     EncodedPixels += EncodedPixels_batch
#     scores=[]
#     for idx in range (masks.shape[2]):
#         scores.append(r['scores'][idx])
#     Scores_all+=scores
#
# print (len(ImageId),len(EncodedPixels),len(Scores_all))
# f.write2csv_score('submission_v12_coco_rgb_final_no_post_baseline.csv', ImageId, EncodedPixels,Scores_all)

#     count = masks.shape[-1]
#     occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
#
#     for i in range(count - 2, -1, -1):
#         mask = masks[:, :, i] * occlusion
#         mask_rle = rle_to_string(rle_encode(mask))
#
#         # Sanity check
#         try:
#             rle_decode(mask_rle, original_image.shape[:-1])
#             output.append([image_id, mask_rle])
#             occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
#
#         except Exception as e:
#             print(e)
#             print(image_id)
#             print('---')
#
# output_df = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels'])
# output_df.to_csv('submission.csv', index=False, encoding='utf-8')
        