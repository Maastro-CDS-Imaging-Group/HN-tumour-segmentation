import numpy as np


def dice(pred_label_volume, target_label_volume):
    """
    Args
        pred_label_volume: Numpy ndarray. Full labelmap volume aggregated from predicted patches. Shape (W,H,D)
        target_label_volume: Numpy ndarray. Full labelmap volume aggregated from GTV patches. Shape (W,H,D)
    Returns:
        dice_score: Dice coefficient
    """
    # To handle cases when target labelmap has zero GTV voxels (there are 2 patients in crS - CHUM010, CHUS021)
    epsilon = 0.001

    intersection = np.sum(pred_label_volume * target_label_volume)
    dice_score = (2 * intersection + epsilon) / (np.sum(pred_label_volume) + np.sum(target_label_volume) + epsilon)
    return dice_score


def jaccard(pred_label_volume, target_label_volume):
    epsilon = 0.001

    intersection = np.sum(pred_label_volume * target_label_volume)
    union = np.sum(np.maximum(pred_label_volume, target_label_volume))
    iou = (intersection + epsilon) / (union + epsilon)
    return iou 