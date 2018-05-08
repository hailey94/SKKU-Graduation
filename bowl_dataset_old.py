from utils import Dataset
from glob import glob
import os
import numpy as np
import re
import cv2
import skimage.color
import skimage.io
from skimage import color
class BowlDataset(Dataset):
    
    
    def load_bowl(self, base_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        self.add_class("bowl", 1, "nuclei")
        
        masks = dict()
        # id_extractor = re.compile(base_path + "\/(?P<image_id>.*)\/masks\/(?P<mask_id>.*)\.png")
        #
        # for mask_path in glob(os.path.join(base_path, "**", "masks", "*.png")):
        #     matches = id_extractor.match(mask_path)
        #
        #     image_id = matches.group("image_id")
        #     image_path = os.path.join(base_path, image_id, "images", image_id + ".png")
        #
        #     if image_path in masks:
        #         masks[image_path].append(mask_path)
        #     else:
        #         masks[image_path] = [mask_path]

        #id_extractor = re.compile(base_path + "\/\.npy")

        for mask_path in glob(os.path.join(base_path, "*.npy")):
            image_id=os.path.splitext(os.path.basename(mask_path))[0]
            image_path = os.path.join(base_path, image_id + ".png")
            if image_path in masks:
                masks[image_path].append(image_path)
            else:
                masks[image_path] = mask_path

        for i, (image_path, mask_paths) in enumerate(masks.items()):
            self.add_image("bowl", image_id=i, path=image_path, mask_paths=mask_paths)


    def load_image(self, image_id):
        info = self.image_info[image_id]

        return cv2.imread(info["path"])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.load(info["mask_paths"])
        class_ids = np.ones(mask.shape[-1])


        return mask, class_ids.astype(np.int32)

    # def load_image(self, image_id):
    #     info = self.image_info[image_id]
    #
    #     image = skimage.io.imread(info["path"])
    #     # RGBA to RGB
    #     if image.shape[2] != 3:
    #         image = image[:, :, :3]
    #     return image
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

            
    # def load_mask(self, image_id):
    #     info = self.image_info[image_id]
    #     mask_paths = info["mask_paths"]
    #     count = len(mask_paths)
    #     #masks = []
    #     masks = [skimage.io.imread(path) for path in mask_paths]
    #     # for i, mask_path in enumerate(mask_paths):
    #     #     masks.append(skimage.io.imread(mask_path))
    #
    #     masks = np.stack(masks, axis=-1)
    #     masks = np.where(masks > 128, 1, 0)
    #
    #     # Handle occlusions
    #     # occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
    #     # for i in range(count-2, -1, -1):
    #     #     masks[:, :, i] = masks[:, :, i] * occlusion
    #     #     occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
    #
    #     class_ids = np.ones(count)
    #     return masks, class_ids.astype(np.int32)


    def load_mask_by_name(self, image_name):
        mask_paths=[]
        for mask_path in glob(os.path.join(image_name, "masks", "*.png")):
            #print (mask_path)
            mask_paths.append(mask_path)
        #print(mask_paths)
        count = len(mask_paths)
        masks = [skimage.io.imread(path) for path in mask_paths]
        # for i, mask_path in enumerate(mask_paths):
        #     masks.append(skimage.io.imread(mask_path))

        masks = np.stack(masks, axis=-1)
        masks = np.where(masks > 128, 1, 0)
        class_ids = np.ones(count)
        return masks, class_ids.astype(np.int32)

    def load_semantic(self, image_id):
        info = self.image_info[image_id]
        mask_paths = info["mask_paths"]
        count = len(mask_paths)
        # masks = []
        import cv2
        masks = [skimage.io.imread(path) for path in mask_paths]

        masks = np.stack(masks, axis=-1)
        masks = np.where(masks > 128, 1, 0)
        print (masks.shape)
        masks=np.sum(masks,axis=2)
        masks[masks > 1] = 1
        return masks

    # def rgb_clahe(self, in_rgb_img):
    #     grid_size = 8
    #     bgr = in_rgb_img[:, :, [2, 1, 0]]  # flip r and b
    #     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
    #     lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    #     bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #     return bgr[:, :, [2, 1, 0]]
