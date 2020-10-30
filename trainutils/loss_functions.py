import torch

CLASS_FREQUENCIES = {0: 199156267, 1: 904661}




def build_loss_function(loss_name, device):
    if loss_name == 'wce':
        ce_weights = torch.Tensor( [
                                    1 - CLASS_FREQUENCIES[0] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1]),
                                    1 - CLASS_FREQUENCIES[1] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1])
                                   ]
                                 ).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=ce_weights, reduction='mean')
    
    elif loss_name == 'wce-and-dice':
      # TODO
      pass
    
    return criterion