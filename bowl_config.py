from config import Config


class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bowl"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 512

    ROI_POSITIVE_RATIO = 0.33
    #
    """
    Number of validation 
    54 10 1
    Number of Training 
    487 97 15
    """
    #
    STEPS_PER_EPOCH = 435 // (IMAGES_PER_GPU * GPU_COUNT)  # 664

    VALIDATION_STEPS = 10 // (IMAGES_PER_GPU * GPU_COUNT)

    LEARNING_RATE = 1e-3

    MAX_GT_INSTANCES = 500

    USE_MINI_MASK = True

    # IMAGE_CROP_WIDTH = 128
    # IMAGE_CROP_HEIGHT = 128


bowl_config = BowlConfig()
bowl_config.display()