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


def jaccard(pred_label_volume, target_label_volume, return_i_and_u=False):
    """
    Jaccard Coefficient (JC) or IoU
    Args
        pred_label_volume: Numpy ndarray. Full labelmap volume aggregated from predicted patches. Shape (D,H,W) or (W,H,D), doesn't matter
        target_label_volume: Numpy ndarray. Full labelmap volume aggregated from GTV patches. Shape (D,H,W) or (W,H,D), doesn't matter
    Returns:
        iou: Intersection-over-Union
    """
    intersection = np.sum(pred_label_volume * target_label_volume).astype(float)
    union = np.sum(np.maximum(pred_label_volume, target_label_volume)).astype(float)
    iou = (intersection + EPSILON) / (union + EPSILON)
    if return_i_and_u:
        return iou, intersection, union    
    else:
        return iou


def hausdorff(pred_label_volume, target_label_volume, dim_ordering='whd'):
    """
    Hausdorff Distance (HD) -- full version (no percentile limit)
    """
    if dim_ordering == 'whd':    spacing = (1,1,3)
    elif dim_ordering == 'dhw':    spacing = (3,1,1)

    # Handling the no-prediction case
    if list(np.unique(pred_label_volume.astype(np.int8))) != [0, 1]:
        return np.NaN

    hausdorff = hd(pred_label_volume, target_label_volume, voxelspacing=spacing, connectivity=1)
    return hausdorff