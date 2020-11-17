import torch
import torch.nn.functional as F


CLASS_FREQUENCIES = {'hecktor-crS_rs113': {0: 199156267, 1: 904661},
                     'hecktor-crFHN_rs113': None
                     } 


class DiceLoss(torch.nn.Module):
  """
  Soft Dice loss
  """
  def __init__(self):
    pass
  
  def forward(self, pred_batch, target_labelmap_batch):
    dice_loss = dice_loss_function(pred_batch, target_labelmap_batch)
    return dice_loss


class CompositeLoss(torch.nn.Module):
  """
  L = L_wce + L_dice

  """
  def __init__(self, dataset_name, device):
    class_frequencies = CLASS_FREQUENCIES[dataset_name]
    ce_weights = torch.Tensor( [
                                1 - class_frequencies[0] / (class_frequencies[0] + class_frequencies[1]),
                                1 - class_frequencies[1] / (class_frequencies[0] + class_frequencies[1])
                                ]
                              )
    self.wce_module = torch.nn.CrossEntropyLoss(weight=ce_weights.to(device), reduction='mean')

  def forward(self, pred_batch, target_labelmap_batch):
    wce_loss = self.wce_module(pred_batch, target_labelmap_batch)
    dice_loss = dice_loss_function(pred_batch, target_labelmap_batch)
    return wce_loss + dice_loss



def dice_loss_function(pred_batch, target_labelmap_batch):
  """
  Soft Dice loss using the dice score of the foreground (GTV)

  Args:
    pred_batch: Batch of predicted scores over patches. Tensor of shape (N,C,D,H,W)
    target_labelmap_batch: Batch of target GTV labelmap patches. Tensor of shape (N,D,H,W)
  Returns:
    dice_loss: 1 - dice
  """
  pred_batch = F.softmax(pred_batch, dim=1) # Convert logits into probabilities
  pred_foreground_batch = pred_batch[:,1,:,:,:] # Only take the foreground probabilities

  batch_dice_loss = 0
  batch_size = target_labelmap_batch.shape[0]
  
  for i in range(batch_size): # Compute dice loss for each sample in the batch

    pred_foreground_sample = pred_foreground_batch[i]
    target_labelmap_sample = target_labelmap_batch[i]

    intersection = torch.sum(pred_foreground_sample * target_labelmap_sample).float()
    sample_dice_score = 2 * intersection / (torch.sum(pred_foreground_sample) + torch.sum(target_labelmap_sample))
    sample_dice_loss = 1 - sample_dice_score
    
    batch_dice_loss += sample_dice_loss
  batch_dice_loss /= batch_size # Take average over the samples
  return batch_dice_loss



def build_loss_function(loss_name, dataset_name, device):
    """
    Loss function builder 
    """
    if loss_name == 'wce':
        class_frequencies = CLASS_FREQUENCIES[dataset_name]
        ce_weights = torch.Tensor( [
                                    1 - class_frequencies[0] / (class_frequencies[0] + class_frequencies[1]),
                                    1 - class_frequencies[1] / (class_frequencies[0] + class_frequencies[1])
                                    ]
                                  )

        wce_module = torch.nn.CrossEntropyLoss(weight=ce_weights.to(device), reduction='mean')
        return wce_module

    elif loss_name == 'dice':
      dice_loss_module = DiceLoss()
      return dice_loss_module


    elif loss_name == 'wce+dice': # Composite of wCE and Dice losses
      composite_loss_module = CompositeLoss(dataset_name, device)
      return composite_loss_module

