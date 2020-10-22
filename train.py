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
from datautils.patch_sampling import PatchSampler3D, PatchQueue, get_num_valid_patches
from datautils.patch_aggregation import PatchAggregator3D, get_pred_labelmap_patches_list
import nnmodules
from trainutils.metrics import volumetric_dice

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

PATCH_SIZE = (128, 128, 32)
FOCAL_POINT_STRIDE = (10, 10, 10)

TRAIN_PATCH_SAMPLER_KWARGS = {'patch_size': PATCH_SIZE, 
                              'sampling': 'random'
                             }
TRAIN_PATCH_QUEUE_KWARGS = {
		                  'max_length': 128,
		                  'samples_per_volume': 32,
		                  'num_workers': 4,
		                  'shuffle_subjects': True,
		                  'shuffle_patches': True
	                     }


VAL_PATCH_SAMPLER_KWARGS = {'patch_size': PATCH_SIZE, 
                              'sampling': 'sequential',
							  'focal_point_stride': FOCAL_POINT_STRIDE
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

CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_LOAD_PATH = "./model_checkpoints/unet3d_pet_9.pt"


# Logging config
logging.basicConfig(level=logging.DEBUG)

USE_WANDB = True

if USE_WANDB:
	wandb.init(entity="cnmy-ro",
			   project="hn-gtv-segmentation",
			   name="training-script-test-run",
			   config={'dataset': "hecktor_train_crS_rs113",
				       'patch_size': PATCH_SIZE,
				       'batch_of_patches_size': BATCH_OF_PATCHES_SIZE,
					   'learning_rate': LEARNING_RATE
					  }
			 )



# -----------------------------------------------
# Safety checks
# -----------------------------------------------

assert PATCH_SIZE[0] % 2**4 == 0 and PATCH_SIZE[1] % 2**4 == 0 and PATCH_SIZE[2] % 2**4 == 0
assert TRAIN_PATCH_QUEUE_KWARGS['max_length'] % TRAIN_PATCH_QUEUE_KWARGS['samples_per_volume'] == 0
assert FOCAL_POINT_STRIDE[0] < PATCH_SIZE[0]/2 and FOCAL_POINT_STRIDE[1] < PATCH_SIZE[1]/2 and FOCAL_POINT_STRIDE[2] < PATCH_SIZE[2]/2


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
valid_patches_per_volume = get_num_valid_patches(PATCH_SIZE, VOLUME_SIZE, focal_point_stride=FOCAL_POINT_STRIDE)
val_sampler = PatchSampler3D(**VAL_PATCH_SAMPLER_KWARGS)
patch_aggregator = PatchAggregator3D(PATCH_SIZE, VOLUME_SIZE, FOCAL_POINT_STRIDE)
val_volume_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# -----------------------------------------------
# Network
# -----------------------------------------------

unet3d = nnmodules.UNet3D(residual=RESIDUAL, normalization=NORMALIZATION).cuda()


# -----------------------------------------------
# Training
# -----------------------------------------------

ce_weights = torch.Tensor( [
	                        1 - CLASS_FREQUENCIES[0] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1]),
	                        1 - CLASS_FREQUENCIES[1] / (CLASS_FREQUENCIES[0] + CLASS_FREQUENCIES[1])
	                       ]
	                     ).cuda()

criterion = torch.nn.CrossEntropyLoss(weight=ce_weights, reduction='mean')

optimizer = torch.optim.Adam(unet3d.parameters(), lr=LEARNING_RATE)


start_epoch = 1
if CONTINUE_FROM_CHECKPOINT:
	unet3d.load_state_dict(torch.load(CHECKPOINT_LOAD_PATH))
	start_epoch = CHECKPOINT_LOAD_PATH.split('.')[-2][-1]
	start_epoch = int(start_epoch) + 1
	print("Continuing from epoch", start_epoch)

if USE_WANDB:
	wandb.watch(unet3d)
	wandb.config.update({'start_epoch': start_epoch,
	                     'epochs': start_epoch + EPOCHS})

# Training loop
for epoch in range(start_epoch, start_epoch+EPOCHS):
	logging.debug(f"Epoch {epoch}")

	epoch_train_loss = 0
	epoch_val_loss = 0

	epoch_val_dice = 0

	# Train
	unet3d.train() # Set the model in train mode
	logging.debug("Training ...")
	for batch_of_patches in tqdm(train_patch_loader):

		PET_patches = batch_of_patches['PET'].cuda()
		GTV_labelmap_patches = batch_of_patches['GTV-labelmap'].long().cuda()

		# Forward pass
		optimizer.zero_grad()
		pred_patches = unet3d(PET_patches)

		# Compute loss
		train_loss = criterion(pred_patches, GTV_labelmap_patches)

		# Generate loss gradients and back-propagate
		train_loss.backward()
		optimizer.step()

		# Calculate and accumulate metrics
		epoch_train_loss += train_loss.item()
		break
	epoch_train_loss /= len(train_patch_loader)

	# Clear CUDA cache
	torch.cuda.empty_cache()


	# Validate
	logging.debug("Validating ...")
	unet3d.eval() # Set the model in inference mode

	# Iterate over patients in validation set
	for patient_dict in tqdm(val_volume_loader):

		# Remove the batch dimension from input and target volumes of the patient dict
		for key, value in patient_dict.items():
			patient_dict[key] = value[0]

		# Get full list of patches
		patches_list = val_sampler.get_samples(patient_dict, num_patches=valid_patches_per_volume)

		patient_pred_patches_list = []

		# Take BATCH_OF_PATCHES_SIZE number of patches at a time and push through the network
		for i in range(0, valid_patches_per_volume, BATCH_OF_PATCHES_SIZE):
			PET_patches = torch.stack([patches_list[i]['PET'] for i in range(BATCH_OF_PATCHES_SIZE)], dim=0).cuda()
			GTV_labelmap_patches = torch.stack([patches_list[i]['GTV-labelmap'] for i in range(BATCH_OF_PATCHES_SIZE)], dim=0).long().cuda()

			with torch.no_grad(): # Disable autograd
				# Forward pass
				pred_patches = unet3d(PET_patches)
				
				# Convert the predicted batch of probabilities to a list of labelmap patches
				patient_pred_patches_list.extend(get_pred_labelmap_patches_list(pred_patches.cpu())) 

				# Compute validation loss
				val_loss = criterion(pred_patches, GTV_labelmap_patches)
				epoch_val_loss += val_loss.item()

		# Aggregate and compute dice
		pred_labelmap_volume = patch_aggregator.aggregate(patient_pred_patches_list) 
		dice_score = volumetric_dice(pred_labelmap_volume, patient_dict['GTV-labelmap'].cpu().numpy())
		break
	epoch_val_loss /= len(val_volume_loader) * valid_patches_per_volume / BATCH_OF_PATCHES_SIZE
	epoch_val_dice /= len(val_volume_loader)
	
	
	# Clear CUDA cache
	torch.cuda.empty_cache()

	# Logging
	logging.debug(f"Training loss: {epoch_train_loss}")
	logging.debug(f"Validation loss: {epoch_val_loss}")
	logging.debug(f"Validation dice: {epoch_val_dice}")
	logging.debug("")
	logging.debug("")

	if USE_WANDB:
		wandb.log({'train-loss': epoch_train_loss,
				   'val-loss': epoch_val_loss,
				   'val-dice': epoch_val_dice
				  },
				  step=epoch)

	# Checkpointing
	if epoch % CHECKPOINT_STEP == 0:
		torch.save(unet3d.state_dict(), f"{CHECK_POINT_DIR}/unet3d_pet_{epoch}.pt")