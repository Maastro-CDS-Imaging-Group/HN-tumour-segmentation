import torch
from torch.nn.functional import interpolate

import nnmodules

backbone_config = {'in_channels': 1, 
                   'out_channels_first_layer': 32, 
                   'num_encoding_blocks': 4, 
                   'residual': True, 
                   'normalization': 'batch' }

attention_module_config = {'in_channels': 1, 
                           'out_channels_first_layer': 32, 
                           'num_encoding_blocks': 4, 
                           'out_classes':1,
                           'residual': True, 
                           'normalization': 'batch' }


msam = nnmodules.MSAM3D(backbone_config, attention_module_config).eval().cuda()

random_pet = torch.randn((1, 1, 32, 128, 128)).cuda()
random_ct = torch.randn((1, 1, 32, 128, 128)).cuda()
output = msam(random_pet, random_ct)
