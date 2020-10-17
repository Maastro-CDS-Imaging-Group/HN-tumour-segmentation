import numpy as np
import torch

from datautils.preprocessing import Preprocessor
from datasets.HECKTORUnimodalityDataset import HECKTORUnimodalityDataset
from datautils.patch_sampling import PatchSampler2D, PatchSampler3D, PatchQueue
import nnmodules


dimensions = 2


data_dir = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
patient_id_filepath = "./hecktor_meta/patient_IDs_train.txt"

'''
# Data pipeline
preprocessor = Preprocessor(
	                        smooth_sigma_mm={'PET': 2.0},
                            standardization_method={'PET': 'clipping'},
                            clipping_range={'PET': [0,20]}
                           )

PET_dataset = HECKTORUnimodalityDataset(
	                                    data_dir,
	                                    patient_id_filepath,
	                                    mode='train',
	                                    preprocessor=preprocessor,
	                                    input_modality='PET',
	                                    augment_data=False
	                                    )

if dimensions == 2:
	sampler = PatchSampler2D(patch_size=(128,128))
elif dimensions == 3:
	sampler = PatchSampler3D(patch_size=(128,128,32))

patch_queue = PatchQueue(
	                     dataset=PET_dataset,
	                     max_length=32,
	                     samples_per_volume=16,
	                     sampler=sampler,
	                     num_workers=0,
	                     shuffle_subjects=True,
	                     shuffle_patches=True
	                     )
'''
# Network
if dimensions == 2:
	unet = nnmodules.UNet2D(residual=True)
elif dimensions == 3:
	unet = nnmodules.UNet3D(residual=True)


# Start testing
if dimensions == 2:
	random_img = torch.rand(1, 1, 128, 128)
elif dimensions == 3:
	random_img = torch.rand(1, 1, 32, 128, 128)

output = unet(random_img)
print(output.shape)