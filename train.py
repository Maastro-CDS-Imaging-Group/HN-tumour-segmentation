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
import wandb

from datautils.preprocessing import Preprocessor
from datasets.hecktor_unimodal_dataset import HECKTORUnimodalDataset
from datautils.patch_sampling import PatchSampler2D, PatchSampler3D, PatchQueue, get_num_valid_patches
import nnmodules
from trainutils.metrics import volumetric_dice

# -----------------------------------------------
# Constants
# -----------------------------------------------

DATA_DIR = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crS_rs113_hecktor_nii"
PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

CLASS_FREQUENCIES = {0: 199156267, 1: 904661}
VOLUME_SIZE = (144, 144, 48)


# -----------------------------------------------
# Configuration settings
# -----------------------------------------------

# Data config
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

DIMENSIONS = 3
PATCH_SIZE_2D = (256, 256)
PATCH_SIZE_3D = (128, 128, 32)

# FOCAL_POINT_STRIDE_2D = np.array(PATCH_SIZE_2D) //  2
# FOCAL_POINT_STRIDE_3D = np.array(PATCH_SIZE_3D) //  2
FOCAL_POINT_STRIDE_3D = (5,5,5)

TRAIN_PATCH_SAMPLER_KWARGS = {'patch_size': PATCH_SIZE_3D, 
                              'sampling': 'random'
                             }
TRAIN_PATCH_QUEUE_KWARGS = {
		                  'max_length': 128,
		                  'samples_per_volume': 32,
		                  'num_workers': 4,
		                  'shuffle_subjects': True,
		                  'shuffle_patches': True
	                     }


VAL_PATCH_SAMPLER_KWARGS = {'patch_size': PATCH_SIZE_3D, 
                              'sampling': 'sequential',
							  'focal_point_stride': FOCAL_POINT_STRIDE_3D
                             }

valid_patches_per_volume = get_num_valid_patches(PATCH_SIZE_3D, VOLUME_SIZE, focal_point_stride=FOCAL_POINT_STRIDE_3D)
VAL_PATCH_QUEUE_KWARGS = {
		                  'max_length': valid_patches_per_volume,
		                  'samples_per_volume': valid_patches_per_volume,
		                  'num_workers': 4,
		                  'shuffle_subjects': False,
		                  'shuffle_patches': False
	                     }


# Network config
RESIDUAL = True
NORMALIZATION = 'batch'   # None or 'batch'


# Training config
BATCH_OF_PATCHES_SIZE = 2
EPOCHS = 12
LEARNING_RATE = 0.0003

CHECKPOINT_STEP = 3
CHECK_POINT_DIR = "./model_checkpoints"

CONTINUE_FROM_CHECKPOINT = False
CHECKPOINT_LOAD_PATH = "./model_checkpoints/unet3d_pet_e10.pt"


# Logging config
logging.basicConfig(level=logging.DEBUG)

USE_WANDB = False

if USE_WANDB:
	wandb.init(entity="cnmy-ro",
			   project="hn-gtv-segmentation",
			   name="training-script-test-run",
			   config={'dataset': "hecktor_train_crS_rs113",
				       'patch_size': PATCH_SIZE_3D,
				       'batch_of_patches_size': BATCH_OF_PATCHES_SIZE,
			   		   'epochs': EPOCHS,
					   'learning_rate': LEARNING_RATE
					  }
			 )


# -----------------------------------------------
# Data pipeline
# -----------------------------------------------

# Datasets
preprocessor = Preprocessor(**PREPROCESSOR_KWARGS)

train_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='training', preprocessor=preprocessor)
val_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='validation', preprocessor=preprocessor)

# Patch samplers
if DIMENSIONS == 2:
	train_sampler = PatchSampler2D(**TRAIN_PATCH_SAMPLER_KWARGS)
	val_sampler = PatchSampler2D(**VAL_PATCH_SAMPLER_KWARGS)
elif DIMENSIONS == 3:
	train_sampler = PatchSampler3D(**TRAIN_PATCH_SAMPLER_KWARGS)
	val_sampler = PatchSampler3D(**VAL_PATCH_SAMPLER_KWARGS)

# Queues and loaders
train_patch_queue = PatchQueue(**TRAIN_PATCH_QUEUE_KWARGS, dataset=train_dataset, sampler=train_sampler)
train_patch_loader = DataLoader(train_patch_queue, batch_size=BATCH_OF_PATCHES_SIZE)

val_patch_queue = PatchQueue(**VAL_PATCH_QUEUE_KWARGS, dataset=val_dataset, sampler=val_sampler)
val_patch_loader = DataLoader(val_patch_queue, batch_size=BATCH_OF_PATCHES_SIZE)


# -----------------------------------------------
# Network
# -----------------------------------------------

if DIMENSIONS == 2:
	unet = nnmodules.UNet2D(residual=RESIDUAL, normalization=NORMALIZATION).cuda()
elif DIMENSIONS == 3:
	unet = nnmodules.UNet3D(residual=RESIDUAL, normalization=NORMALIZATION).cuda()


# -----------------------------------------------
# Training
# -----------------------------------------------

ce_weights = torch.Tensor( [
	                        1 - CLASS_FREQUENCIES[0] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1]),
	                        1 - CLASS_FREQUENCIES[1] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1])
	                       ]
	                     ).cuda()

criterion = torch.nn.CrossEntropyLoss(weight=ce_weights, reduction='mean')

optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

# Training loop
if USE_WANDB:
	wandb.watch(unet)

start_epoch = 1
if CONTINUE_FROM_CHECKPOINT:
	unet.load_state_dict(torch.load(CHECKPOINT_LOAD_PATH))
	start_epoch = CHECKPOINT_LOAD_PATH.split('.')[0][-2]
	start_epoch = int(start_epoch) + 1

for epoch in range(start_epoch, start_epoch+EPOCHS):
	logging.debug(f"Epoch {epoch}")

	epoch_train_loss = 0
	epoch_val_loss = 0

	epoch_train_dice = 0
	epoch_val_dice = 0

	# Train
	unet.train() # Set the model in train mode
	logging.debug("Training ...")
	for batch_of_patches in tqdm(train_patch_loader):

		PET_patches = batch_of_patches['PET'].cuda()
		GTV_labelmap_patches = batch_of_patches['GTV-labelmap'].long().cuda()

		# Forward pass
		optimizer.zero_grad()
		pred_patches = unet(PET_patches)

		# Compute loss
		train_loss = criterion(pred_patches, GTV_labelmap_patches)

		# Generate loss gradients and back-propagate
		train_loss.backward()
		optimizer.step()

		# Calculate and accumulate metrics
		epoch_train_loss += train_loss.item()
		dice_score = volumetric_dice(pred_patches.detach().cpu(), GTV_labelmap_patches.cpu())
		epoch_train_dice += dice_score
		break

	epoch_train_loss /= len(train_patch_loader)
	epoch_train_dice /= len(train_patch_loader)

	# Clear CUDA cache
	torch.cuda.empty_cache()


	# Validate
	logging.debug("Validating ...")
	unet.eval() # Set the model in inference mode
	for batch_of_patches in tqdm(val_patch_loader):

		PET_patches = batch_of_patches['PET'].cuda()
		GTV_labelmap_patches = batch_of_patches['GTV-labelmap'].long().cuda()

		with torch.no_grad(): # Disable autograd
			# Forward pass
			pred_patches = unet(PET_patches)

			# Compute validation loss
			val_loss = criterion(pred_patches, GTV_labelmap_patches)
			epoch_val_loss += val_loss.item()

			# Calculate and accumulate metrics
			dice_score = volumetric_dice(pred_patches.cpu(), GTV_labelmap_patches.cpu())
			epoch_val_dice += dice_score
			break
	epoch_val_loss /= len(val_patch_loader)
	epoch_val_dice /= len(val_patch_loader)


	# Clear CUDA cache
	torch.cuda.empty_cache()

	# Logging
	logging.debug(f"Training loss: {epoch_train_loss}")
	logging.debug(f"Validation loss: {epoch_val_loss}")
	logging.debug(f"Training dice: {epoch_train_dice}")
	logging.debug(f"Validation dice: {epoch_val_dice}")
	logging.debug("")
	logging.debug("")

	if USE_WANDB:
		wandb.log({'train-loss': epoch_train_loss,
				   'val-loss': epoch_val_loss,
				   'train-dice': epoch_train_dice,
				   'val-dice': epoch_val_dice
				  })

	# Checkpointing
	if epoch % CHECKPOINT_STEP == 0:
		torch.save(unet.state_dict(), f"{CHECK_POINT_DIR}/unet3d_pet_{epoch}.pt")