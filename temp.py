import torch
from torch.nn.functional import interpolate

import nnmodules

# backbone_config = {'in_channels': 1, 
#                    'out_channels_first_layer': 32, 
#                    'num_encoding_blocks': 4, 
#                    'residual': True, 
#                    'normalization': 'batch' }

# attention_module_config = {'in_channels': 1, 
#                            'out_channels_first_layer': 32, 
#                            'num_encoding_blocks': 4, 
#                            'out_classes':1,
#                            'residual': True, 
#                            'normalization': 'batch' }


# msam = nnmodules.MSAM3D(backbone_config, attention_module_config).eval().cuda()

# random_pet = torch.randn((1, 1, 32, 128, 128)).cuda()
# random_ct = torch.randn((1, 1, 32, 128, 128)).cuda()
# output = msam(random_pet, random_ct)

a = torch.tensor([1,2,3], device='cuda')

optimizer = torch.optim.SGD([a], lr=0.1)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                            base_lr=0.1, max_lr=1,
                                            step_size_up=10,
                                            mode='triangular2')

for i in range(1000):
    print(scheduler.get_lr())  
    scheduler.step()
                                          