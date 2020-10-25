"""
Tests to perform: 

1. Whether the training mechanism is working or not - Does the train loss decrease over epochs? -- Pass
2. If the network really getting good - Does the validation loss decrease over epochs ?


Current test: 2
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
from trainer import Trainer


# -----------------------------------------------
# Constants
# -----------------------------------------------

DATASET_NAME = "hecktor-crS_rs113"
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
TRAIN_FOCAL_POINT_STRIDE = (4,4,4) # Anything above 4, and the patches do not cover the volume completely
VAL_FOCAL_POINT_STRIDE = (60, 60, 20) # Sparser focalpoints, but padding ensures that the volume is covered completely
VAL_PRE_SAMPLE_PADDING = (44, 44, 4) # Padding to ensure the number of patches is 8
BATCH_OF_PATCHES_SIZE = 2
val_valid_patches_per_volume = get_num_valid_patches(PATCH_SIZE, 
                                                     VOLUME_SIZE, 
													 focal_point_stride=VAL_FOCAL_POINT_STRIDE,
													 padding=VAL_PRE_SAMPLE_PADDING)
print("val_valid_patches_per_volume:", val_valid_patches_per_volume)

TRAIN_PATCH_SAMPLER_KWARGS = {
	                          'patch_size': PATCH_SIZE, 
							  'volume_size': VOLUME_SIZE,
                              'sampling': 'strided-random',
							  'focal_point_stride': TRAIN_FOCAL_POINT_STRIDE
                             }
TRAIN_PATCH_QUEUE_KWARGS = {
		                  'max_length': 128,
		                  'samples_per_volume': 32,
		                  'num_workers': 4,
		                  'shuffle_subjects': True,
		                  'shuffle_patches': True
	                     }


VAL_PATCH_SAMPLER_KWARGS = {
	                         'patch_size': PATCH_SIZE, 
							 'volume_size': VOLUME_SIZE,
                             'sampling': 'sequential',
							 'focal_point_stride': VAL_FOCAL_POINT_STRIDE,
							 'padding': VAL_PRE_SAMPLE_PADDING
                           }

VAL_AGGREGATOR_KWARGS = {
	                    'patch_size': PATCH_SIZE,
						'volume_size': VOLUME_SIZE,
						'focal_point_stride': VAL_FOCAL_POINT_STRIDE,
						'overlap_handling': 'union',
						'unpadding': VAL_PRE_SAMPLE_PADDING
                        }

# Network configuration --
RESIDUAL = True
NORMALIZATION = 'batch'   # None or 'batch'


# Training configuration -- 
INPUT_DATA_CONFIG = {
	                 'is-bimodal': False,
	                 'input-modality': 'PET', 
                     'input-representation': None
					 }

TRAINING_CONFIG = {'dataset-name': DATASET_NAME,
	               'subset-name': 'crossval-CHUM-training',
	               'loss-name': 'weighted-cross-entropy', 
                   'num-epochs': 12,
				   'learning-rate': 0.0003,
				   'enable-checkpointing': False,
				   'checkpoint-step': 3,
				   'checkpoint-dir': "./model_checkpoints",
				   'continue-from-checkpoint': True,
				   'checkpoint-filename': "unet3d_pet_e021.pt"
				   }

VALIDATION_CONFIG = {'subset-name': 'crossval-CHUM-validation',
	                 'batch-of-patches-size': BATCH_OF_PATCHES_SIZE,
                     'valid-patches-per-volume': val_valid_patches_per_volume
					 }


WANDB_CONFIG = {
	            'patch-size': PATCH_SIZE
		 	   }
LOGGING_CONFIG = {'enable-wandb': False,
                   'wandb-entity': "cnmy-ro",
				   'wandb-project': 'hn-gtv-segmentation',
				   'wandb-run-name': "training-script-test-run",
				   'wandb-config': WANDB_CONFIG
				 }



def main():
	# -----------------------------------------------
	# Safety checks
	# -----------------------------------------------

	assert PATCH_SIZE[0] % 2**4 == 0 and PATCH_SIZE[1] % 2**4 == 0 and PATCH_SIZE[2] % 2**4 == 0
	assert TRAIN_PATCH_QUEUE_KWARGS['max_length'] % TRAIN_PATCH_QUEUE_KWARGS['samples_per_volume'] == 0
	assert val_valid_patches_per_volume % BATCH_OF_PATCHES_SIZE == 0


	# -----------------------------------------------
	# Data pipeline
	# -----------------------------------------------

	# Datasets
	preprocessor = Preprocessor(**PREPROCESSOR_KWARGS)
	train_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='training', preprocessor=preprocessor)
	val_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='validation', preprocessor=preprocessor)

	# Patch based training stuff
	train_sampler = PatchSampler3D(**TRAIN_PATCH_SAMPLER_KWARGS)
	train_patch_queue = PatchQueue(**TRAIN_PATCH_QUEUE_KWARGS, dataset=train_dataset, sampler=train_sampler)
	train_patch_loader = DataLoader(train_patch_queue, batch_size=BATCH_OF_PATCHES_SIZE)

	# Patch based inference stuff
	val_sampler = PatchSampler3D(**VAL_PATCH_SAMPLER_KWARGS)
	val_aggregator = PatchAggregator3D(**VAL_AGGREGATOR_KWARGS)
	val_volume_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


	# -----------------------------------------------
	# Network
	# -----------------------------------------------

	unet3d = nnmodules.UNet3D(residual=RESIDUAL, normalization=NORMALIZATION).to(DEVICE)


	# -----------------------------------------------
	# Training
	# -----------------------------------------------

	trainer = Trainer(unet3d, 
					train_patch_loader, val_volume_loader, val_sampler, val_aggregator,
					DEVICE,
					INPUT_DATA_CONFIG, TRAINING_CONFIG, VALIDATION_CONFIG, LOGGING_CONFIG)

	trainer.run_training()


if __name__ == '__main__':
	main()