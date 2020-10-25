"""
Tests to perform: 

1. ...

"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datautils.preprocessing import Preprocessor
from datasets.hecktor_unimodal_dataset import HECKTORUnimodalDataset
from datautils.patch_sampling import PatchSampler3D, PatchQueue, get_num_valid_patches
from datautils.patch_aggregation import PatchAggregator3D, get_pred_labelmap_patches_list
import nnmodules
from inferer import Inferer


# -----------------------------------------------
# Constants
# -----------------------------------------------

DATA_DIR = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crS_rs113_hecktor_nii"
PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

CLASS_FREQUENCIES = {0: 199156267, 1: 904661}
VOLUME_SIZE = (144, 144, 48)  # (W,H,D)


# -----------------------------------------------
# Configuration settings
# -----------------------------------------------

# Hardware --
DEVICE = 'cuda'


# Data configuration --
DATASET_KWARGS = {
                'data_dir': DATA_DIR,
                'patient_id_filepath': PATIENT_ID_FILEPATH,
                'input_modality': 'PET',
                'augment_data': False
                }

PREPROCESSOR_KWARGS = {
                     'smooth_sigma_mm': {'PET': 2.0, 'CT': None},
                     'standardization_method': {'PET': 'clipping', 'CT': None},
                     'clipping_range': {'PET': [0 ,20], 'CT': None}
                     }

PATCH_SIZE = (128, 128, 32)

FOCAL_POINT_STRIDE = (60, 60, 20) # Sparser focalpoints, but padding ensures that the volume is covered completely
PRE_SAMPLE_PADDING = (44, 44, 4)  # Padding to ensure the number of patches is 8
BATCH_OF_PATCHES_SIZE = 8
valid_patches_per_volume = get_num_valid_patches(PATCH_SIZE, 
                                                     VOLUME_SIZE, 
													 focal_point_stride=FOCAL_POINT_STRIDE,
													 padding=PRE_SAMPLE_PADDING)
print("valid_patches_per_volume:", valid_patches_per_volume)


PATCH_SAMPLER_KWARGS = {
	                         'patch_size': PATCH_SIZE, 
							 'volume_size': VOLUME_SIZE,
                             'sampling': 'sequential',
							 'focal_point_stride': FOCAL_POINT_STRIDE,
							 'padding': PRE_SAMPLE_PADDING
                           }

PATCH_AGGREGATOR_KWARGS = {
	                    'patch_size': PATCH_SIZE,
						'volume_size': VOLUME_SIZE,
						'focal_point_stride': FOCAL_POINT_STRIDE,
						'overlap_handling': 'union',
						'unpadding': PRE_SAMPLE_PADDING
                        }

# Network configuration --
RESIDUAL = True
NORMALIZATION = 'batch'   # None or 'batch'


# Training configuration -- 
INPUT_DATA_CONFIG = {'is-bimodal': False,
	                 'input-modality': 'PET', 
                     'input-representation': None
                     } 				   

INFERENCE_CONFIG = { 'dataset-name': 'hecktor-crS_rs113-CHUM',
                     'patient-id-filepath': PATIENT_ID_FILEPATH,
                     'batch-of-patches-size': BATCH_OF_PATCHES_SIZE,
                     'valid-patches-per-volume': valid_patches_per_volume,
                     'model-filepath': './saved_models/unet3d_pet_e030.pt',
                     'save-nrrd': True,
                     'output-save-dir': '/home/zk315372/Chinmay/hn_experiment_results/hecktor-crS_rs113'
                     }



# -----------------------------------------------
# Safety checks
# -----------------------------------------------

assert PATCH_SIZE[0] % 2**4 == 0 and PATCH_SIZE[1] % 2**4 == 0 and PATCH_SIZE[2] % 2**4 == 0
assert valid_patches_per_volume % BATCH_OF_PATCHES_SIZE == 0


# -----------------------------------------------
# Data pipeline
# -----------------------------------------------

# Dataset
preprocessor = Preprocessor(**PREPROCESSOR_KWARGS)
inference_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='validation', preprocessor=preprocessor)

# Patch based inference stuff
volume_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
patch_sampler = PatchSampler3D(**PATCH_SAMPLER_KWARGS)
patch_aggregator = PatchAggregator3D(**PATCH_AGGREGATOR_KWARGS)


# -----------------------------------------------
# Network
# -----------------------------------------------

unet3d = nnmodules.UNet3D(residual=RESIDUAL, normalization=NORMALIZATION).to(DEVICE)


# -----------------------------------------------
# Training
# -----------------------------------------------

inferer = Inferer(unet3d, 
                  volume_loader, patch_sampler, patch_aggregator,
                  DEVICE,
                  INPUT_DATA_CONFIG,
                  INFERENCE_CONFIG)

inferer.run_inference()