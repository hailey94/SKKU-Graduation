import numpy as np
import pandas as pd
from skimage import morphology
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


def remove_conn_components(pred_mask, num_cc):
    labels = label(pred_mask)

    if num_cc == 1:

        maxArea = 0
        for region in regionprops(labels):
            if region.area > maxArea:
                maxArea = region.area
                print(maxArea)

        mask = remove_small_objects(labels, maxArea - 1)

    else:
        mask = remove_small_objects(labels, 50, connectivity=2)

    return mask


def fillhole(masks):
    _, _, num_masks = masks.shape
    masks_post = []
    for i in range(num_masks):
        mask = masks[:, :, i]
        mask = binary_fill_holes(mask)
        masks_post.append(mask)
    masks_post = np.stack(masks_post, axis=-1)
    masks_post = np.array(masks_post, dtype=np.uint8)
    return masks_post


def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths


def numpy2encoding(predicts, img_name):
    """predicts: [H, W, N] instance binary masks
    注意： 中间可能有洞
    """
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]):
        rle = run_length_encoding(predicts[:, :, i])
        ImageId.append(img_name)
        EncodedPixels.append(rle)
    return ImageId, EncodedPixels


def numpy2encoding_no_overlap(predicts, img_name):
    """predicts: [H, W, N] instance binary masks
    注意： 中间可能有洞
    remove overlapping parts
    """
    sum_predicts = np.sum(predicts, axis=2)
    sum_predicts[sum_predicts >= 2] = 0
    sum_predicts = np.expand_dims(sum_predicts, axis=-1)
    predicts = predicts * sum_predicts

    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]):
        rle = run_length_encoding(predicts[:, :, i])
        if len(rle) > 0:
            ImageId.append(img_name)
            EncodedPixels.append(rle)
    return ImageId, EncodedPixels


def numpy2encoding_no_overlap2(predicts, img_name, scores):
    """predicts: [H, W, N] instance binary masks
    注意： 中间可能有洞
    overlapping parts are given to the instance of highest score (i.e. DETECTION CONFIDENCE)
    """
    # ====refine your masks here !=====
    predicts = filter_small(predicts, 20)
    # for i in range(predicts.shape[2]):
    #     predicts[:,:,i] = remove_conn_components(predicts[:,:,i],1)
    # for i in range(predicts.shape[2]):
    #     predicts[:, :, i] = remove_small_objects(predicts[:, :, i], 20, connectivity=2)

    for i in range(predicts.shape[2]):
        predicts[:, :, i] = refineMasks(predicts[:, :, i])
    predicts = fillhole(predicts)
    # ==========================
    sum_predicts = np.sum(predicts, axis=2)
    rows, cols = np.where(sum_predicts >= 2)

    for i in zip(rows, cols):
        instance_indicies = np.where(np.any(predicts[i[0], i[1], :]))[0]
        highest = instance_indicies[0]
        predicts[i[0], i[1], :] = predicts[i[0], i[1], :] * 0
        predicts[i[0], i[1], highest] = 1
    # predicts=clean_overlap(predicts,scores, predicts.shape[0], predicts.shape[1])
    predicts = filter_small(predicts, 40)  # best40
    # for i in range(predicts.shape[2]):
    #     predicts[:,:,i] = remove_conn_components(predicts[:,:,i],2)
    # for i in range(predicts.shape[2]):
    #     predicts[:, :, i] = remove_small_objects(predicts[:, :, i], 40, connectivity=2)
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]):
        rle = run_length_encoding(predicts[:, :, i])
        if len(rle) > 0:
            ImageId.append(img_name)
            EncodedPixels.append(rle)
    return ImageId, EncodedPixels


def refineMasks(mask):
    import cv2
    kernel = np.ones((2, 2), np.uint8)
    masks_post = cv2.dilate(mask, kernel, iterations=1)
    return masks_post
    # return binary_dilation(mask, disk(1))


def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({'ImageId': ImageId, 'EncodedPixels': EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])


def write2csv_score(file, ImageId, EncodedPixels, Scores):
    df = pd.DataFrame({'ImageId': ImageId, 'EncodedPixels': EncodedPixels, 'Scores': Scores})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels', 'Scores'])


def clean_img(x):
    """http://blog.csdn.net/haoji007/article/details/52063306
        closing 先膨胀再腐蚀，可用来填充孔洞
        opening 先腐蚀再膨胀，可以消除小物体或小斑块
    https://www.kaggle.com/kmader/nuclei-overview-to-submission:
        remove single pixels, connect nearby regions
    """
    return binary_opening(binary_closing(x, disk(1)), disk(3))


## post process #######################################################################################
def filter_small(masks, threshold):
    _, _, num_masks = masks.shape
    masks_post = []
    for i in range(num_masks):
        mask_small = masks[:, :, i]
        if (mask_small.sum() > threshold):
            masks_post.append(mask_small)
    masks_post = np.stack(masks_post, axis=-1)
    return masks_post


def clean_overlap(masks, scores, height, width):
    if masks.shape[0] == 0:
        masks = np.zeros([height, width, 1])
        masks[0, 0, 0] = 1

    masks = np.moveaxis(masks, [0, 1, 2], [1, 2, 0])
    sort_ind = np.argsort(scores)[::-1]
    masks = masks[sort_ind]
    overlap = np.zeros([height, width])
    for mm in range(len(masks)):
        mask = masks[mm]
        overlap += mask
        mask[overlap > 1] = 0
        masks[mm] = mask

    del_ind = np.where(np.sum(masks, axis=(1, 2)) < 1)[0]
    if len(del_ind) > 0:
        if len(del_ind) < len(masks):
            print('Empty mask, deleting', len(del_ind), 'masks')
            masks = np.delete(masks, del_ind, axis=0)
        else:
            masks = np.zeros([1, height, width])
            masks[0, 0, 0] = 1

    return masks

