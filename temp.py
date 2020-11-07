import numpy as np
import torch
from trainutils.loss_functions import *
from inferutils.metrics import volumetric_dice

# pred = np.array([
#                  [[0.0, 1.0],
#                   [0.0, 1.0]],
                
#                  [[1.0, 0.0],
#                   [1.0, 0.0]]
#                 ])

pred = np.array([
                 [[0.0, 1.0],
                  [1.0, 0.0]],
                
                 [[1.0, 0.0],
                  [0.0, 1.0]]
                ])                

pred = torch.tensor(pred, requires_grad=True)
print(pred.is_leaf) 

target_labelmap = torch.tensor(([[0, 1],
                                 [0, 1]]))

print(pred.shape, target_labelmap.shape)

criterion = build_loss_function('dice', device='cpu')
dice_loss = criterion(pred.unsqueeze(dim=0), target_labelmap.unsqueeze(dim=0))
dice_loss.backward()
print(dice_loss)

print(volumetric_dice(pred.detach().argmax(dim=1).numpy(), target_labelmap.numpy()))
print(pred.argmax(dim=1))