import numpy as np


def volumetric_dice(pred_label_volume, target_label_volume):
    """
    Args
        pred_label_volume: Full labelmap volume aggregated from predicted patches. shape (W,H,D)
        target_label_volume: Full labelmap volume aggregated from GTV patches. shape (W,H,D)
    """
    epsilon = 1e-4 # To deal with cases when target labelmap has zero GTV voxels (there are 2 patients in crS - CHUM010, CHUS021)
    intersection = np.sum(pred_label_volume * target_label_volume)
    dice_score = 2 * intersection / (np.sum(pred_label_volume) + np.sum(target_label_volume) + epsilon)
    return dice_score


def iou(pred_tensor, gt_label_tensor):
    pass
