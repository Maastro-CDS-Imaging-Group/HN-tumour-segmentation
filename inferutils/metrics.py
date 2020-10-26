import numpy as np


def volumetric_dice(pred_label_volume, gtv_label_volume):
    """
    pred_label_volume: Full labelmap volume aggregated from predicted patches. shape (W,H,D)
    gtv_label_volume: Full labelmap volume aggregated from GTV patches. shape (W,H,D)
    """

    intersection = np.sum(pred_label_volume * gtv_label_volume)
    dice_score = 2 * intersection / (np.sum(pred_label_volume) + np.sum(gtv_label_volume))
    return dice_score


def iou(pred_tensor, gt_label_tensor):
    pass
