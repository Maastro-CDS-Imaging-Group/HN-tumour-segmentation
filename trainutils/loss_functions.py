import numpy as np
import torch
import torch.nn.functional as F


CLASS_FREQUENCIES = {'hecktor-crS_rs113': {0: 199156267, 1: 904661},
                     'hecktor-crFHN_rs113': None
                     } 

BATCHES_DIM = 0
CHANNELS_DIM = 1
SPATIAL_DIMS = (2,3,4)
EPSILON = 0.0001


class DiceLoss(torch.nn.Module):
  """
  Soft Dice loss, using the dice score of the foreground (GTV)

  Args:
    pred_batch: Batch of predicted scores over patches. Tensor of shape (N,C,D,H,W)
    target_labelmap_batch: Batch of target GTV labelmap patches. Tensor of shape (N,D,H,W)
  Returns:
    dice_loss: 1 - dice
  """
  def __init__(self):
    super().__init__()
    
  
  def forward(self, pred_batch, target_labelmap_batch):
    pred_batch = F.softmax(pred_batch, dim=CHANNELS_DIM) # Convert logits into probabilities
    pred_foreground_batch = pred_batch[:,1,:,:,:] # Only take the foreground probabilities. Shape (N,D,H,W)

    target_labelmap_batch = target_labelmap_batch.float()

    spatial_dims = tuple(np.array(SPATIAL_DIMS) - 1)
    batch_size = target_labelmap_batch.shape[BATCHES_DIM]
    
    intersection = torch.sum(pred_foreground_batch * target_labelmap_batch, dim=spatial_dims)
    score = (2.0 * intersection + EPSILON) / \
            (torch.sum(pred_foreground_batch, dim=spatial_dims) + torch.sum(target_labelmap_batch, dim=spatial_dims) + EPSILON)
    
    score = score.sum() / batch_size
    dice_loss = 1 - score
    return dice_loss


class WCEDiceCompositeLoss(torch.nn.Module):
  """
  L = L_wce + L_dice
  """
  def __init__(self, dataset_name, device):
    super().__init__()
    class_frequencies = CLASS_FREQUENCIES[dataset_name]
    ce_weights = torch.Tensor( [
                                1 - class_frequencies[0] / (class_frequencies[0] + class_frequencies[1]),
                                1 - class_frequencies[1] / (class_frequencies[0] + class_frequencies[1])
                                ]
                              )
    self.wce_module = torch.nn.CrossEntropyLoss(weight=ce_weights.to(device), reduction='mean')
    self.dice_loss_module = DiceLoss()

  def forward(self, pred_batch, target_labelmap_batch):
    wce_loss = self.wce_module(pred_batch, target_labelmap_batch)
    dice_loss = self.dice_loss_module(pred_batch, target_labelmap_batch)
    return wce_loss + dice_loss



def build_loss_function(loss_name, dataset_name, device):
    """
    Loss function builder 
    """
    if loss_name == 'wce':
        class_frequencies = CLASS_FREQUENCIES[dataset_name]
        ce_weights = torch.Tensor( [
                                    1 - class_frequencies[0] / (class_frequencies[0] + class_frequencies[1]),
                                    1 - class_frequencies[1] / (class_frequencies[0] + class_frequencies[1])
                                   ],
                                  device=device
                                 )
        wce_module = torch.nn.CrossEntropyLoss(weight=ce_weights, reduction='mean')
        return wce_module

    elif loss_name == 'dice': # Soft Dice loss
      dice_loss_module = DiceLoss()
      return dice_loss_module

    elif loss_name == 'wce+dice': # Composite of wCE and Dice losses
      composite_loss_module = WCEDiceCompositeLoss(dataset_name, device)
      return composite_loss_module

