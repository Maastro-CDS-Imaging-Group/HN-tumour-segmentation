import numpy as np

def volumetric_dice(pred_tensor, gt_label_tensor):
    """
    pred_tensor: shape (N,C,D,H,W)
    gt_label_tensor: shape (N,D,H,W)
    """
    pred_label_np = np.argmax(pred_tensor.numpy(), axis=1)
    gt_label_np = gt_label_tensor.numpy()

    intersection = np.sum(pred_label_np * gt_label_np)
    dice_score = 2 * intersection / (np.sum(pred_label_np) + np.sum(gt_label_np))
    return dice_score


def iou(pred_tensor, gt_label_tensor):
    pred_label_np = np.argmax(pred_tensor.numpy(), axis=1)
    gt_label_np = gt_label_tensor.numpy()

    intersection = np.sum(pred_label_np * gt_label_np)
    union = np.sum(np.maximum(pred_label_np, gt_label_np))
    return intersection / union
