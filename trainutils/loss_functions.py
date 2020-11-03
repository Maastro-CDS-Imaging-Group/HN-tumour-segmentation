import torch
import torch.nn.functional as F


CLASS_FREQUENCIES = {0: 199156267, 1: 904661} # For hecktor-crS_rs113 dataset version

CE_WEIGHTS = torch.Tensor( [
                            1 - CLASS_FREQUENCIES[0] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1]),
                            1 - CLASS_FREQUENCIES[1] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1])
                            ]
                          )


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
    


def build_loss_function(loss_name, device):
    if loss_name == 'wce':
        return torch.nn.CrossEntropyLoss(weight=CE_WEIGHTS.to(device), reduction='mean')


    elif loss_name == 'dice':
      return dice_loss_function


    elif loss_name == 'wce+dice': # Composite of wCE and Dice losses
      
      def criterion(pred_batch, target_labelmap_batch):
        wce_loss = F.cross_entropy(pred_batch, target_labelmap_batch, weight=CE_WEIGHTS.to(device), reduction='mean')
        dice_loss = dice_loss_function(pred_batch, target_labelmap_batch)
        return wce_loss + dice_loss
      
      return criterion

