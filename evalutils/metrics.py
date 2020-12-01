import numpy as np
from medpy.metric.binary import hd


# To prevent 0/0, in cases where target labelmap has zero GTV voxels (there are 2 patients in crS - CHUM010, CHUS021)
EPSILON = 0.0001


def dice(pred_label_volume, target_label_volume):
    """
    Dice Similarity Coefficient (DSC)
    Args
        pred_label_volume: Numpy ndarray. Full labelmap volume aggregated from predicted patches. Shape (D,H,W) or (W,H,D), doesn't matter
        target_label_volume: Numpy ndarray. Full labelmap volume aggregated from GTV patches. Shape (D,H,W) or (W,H,D), doesn't matter
    Returns:
        dice_score: Dice coefficient
    """
    intersection = np.sum(pred_label_volume * target_label_volume)
    dice_score = (2 * intersection + EPSILON) / (np.sum(pred_label_volume) + np.sum(target_label_volume) + EPSILON)
    return dice_score


def jaccard(pred_label_volume, target_label_volume):
    """
    Jaccard Coefficient (JC) or IoU
    Args
        pred_label_volume: Numpy ndarray. Full labelmap volume aggregated from predicted patches. Shape (D,H,W) or (W,H,D), doesn't matter
        target_label_volume: Numpy ndarray. Full labelmap volume aggregated from GTV patches. Shape (D,H,W) or (W,H,D), doesn't matter
    Returns:
        iou: Intersection-over-Union
    """
    intersection = np.sum(pred_label_volume * target_label_volume)
    union = np.sum(np.maximum(pred_label_volume, target_label_volume))
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou 


def hausdorff_distance(pred_label_volume, target_label_volume, dim_ordering='dhw'):
    """
    Hausdorff Distance (HD) -- full version
    """
    # TODO Benchmark and compare Medpy's implementation with Deepmind's
    if dim_ordering == 'whd':    spacing = (1,1,3)
    elif dim_ordering == 'dhw':    spacing = (3,1,1)
    hausdorff = hd(pred_label_volume, target_label_volume, voxelspacing=spacing, connectivity=1)
    return hausdorff