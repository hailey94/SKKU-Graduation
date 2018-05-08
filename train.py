import os
import sys
from bowl_config import bowl_config
from bowl_dataset import BowlDataset
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

model = modellib.MaskRCNN(mode="training", config=bowl_config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
    # model.load_weights('./resnet152_weights_tf.h5', by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "densenet":
    model.load_weights(model.get_imagenet_densenet_weight(), by_name=True)

elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Training dataset
dataset_train = BowlDataset()
# dataset_train.load_bowl('kaggle-dsbowl-2018-dataset-fixes/stage1_train')
dataset_train.load_bowl('stage1_train/mosaics')
dataset_train.prepare()

# # Validation dataset
dataset_val = BowlDataset()
dataset_val.load_bowl('stage1_train/mosaics_val')
dataset_val.prepare()


# Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE / 10,  # Best 0.44 la SGD ko chia 10
            epochs=60,
            layers='heads')

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE / 10,
            epochs=80,
            layers='all')

print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE / 100,
            epochs=100,
            layers='all')

print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE / 1000,
            epochs=130,
            layers='all')
