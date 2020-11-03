import numpy as np


def volumetric_dice(pred_label_volume, target_label_volume):
    """
    Args
        pred_label_volume: Numpy ndarray. Full labelmap volume aggregated from predicted patches. Shape (W,H,D)
        target_label_volume: Numpy ndarray. Full labelmap volume aggregated from GTV patches. Shape (W,H,D)
    Returns:
        dice_score: Dice coefficient
    """
    # Deal with cases when target labelmap has zero GTV voxels (there are 2 patients in crS - CHUM010, CHUS021)
    if 1 not in np.unique(pred_label_volume) and 1 not in np.unique(target_label_volume):
        return 0
    
    intersection = np.sum(pred_label_volume * target_label_volume)
    dice_score = 2 * intersection / (np.sum(pred_label_volume) + np.sum(target_label_volume))
    return dice_score


def jaccard_index(pred_label_volume, target_label_volume):
    intersection = np.sum(pred_label_volume * target_label_volume)
    union = np.sum(np.maximum(pred_label_volume, target_label_volume))
    iou = intersection/union
    return iou 