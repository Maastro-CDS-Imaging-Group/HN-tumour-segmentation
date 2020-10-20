import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datautils.preprocessing import Preprocessor
from datasets.hecktor_unimodal_dataset import HECKTORUnimodalDataset
from datautils.patch_sampling import PatchSampler2D, PatchSampler3D, PatchQueue, get_num_valid_patches
import nnmodules


# -----------------------------------------------
# Constants
# -----------------------------------------------

CLASS_FREQUENCIES = {0: 4069313577, 1: 936423}


# -----------------------------------------------
# Configuration settings
# -----------------------------------------------

logging.basicConfig(level=logging.DEBUG)


# Data config
DATA_DIR = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

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

TRAIN_PATCH_QUEUE_KWARGS = {
		                  'max_length': 128,
		                  'samples_per_volume': 32,
		                  'num_workers': 4,
		                  'shuffle_subjects': True,
		                  'shuffle_patches': True
	                     }

VAL_PATCH_QUEUE_KWARGS = {
		                  'max_length': get_num_valid_patches(PATCH_SIZE_2D),
		                  'samples_per_volume': get_num_valid_patches(PATCH_SIZE_2D),
		                  'num_workers': 4,
		                  'shuffle_subjects': False,
		                  'shuffle_patches': False
	                     }



# Network config
RESIDUAL = True
NORMALIZATION = None   # None or 'batch'


# Training config
BATCH_OF_PATCHES_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001


# -----------------------------------------------
# Data pipeline
# -----------------------------------------------

# Datasets
preprocessor = Preprocessor(**PREPROCESSOR_KWARGS)

train_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='training', preprocessor=preprocessor)
val_dataset = HECKTORUnimodalDataset(**DATASET_KWARGS, mode='validation', preprocessor=preprocessor)

# Patch samplers
if DIMENSIONS == 2:
	train_sampler = PatchSampler2D(patch_size=PATCH_SIZE_2D, sampling='random')
	val_sampler = PatchSampler2D(patch_size=PATCH_SIZE_2D, sampling='sequential')
elif DIMENSIONS == 3:
	train_sampler = PatchSampler3D(patch_size=PATCH_SIZE_3D, sampling='random')
	val_sampler = PatchSampler3D(patch_size=PATCH_SIZE_3D, sampling='sequential')

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
for epoch in range(EPOCHS):

	running_train_loss = 0
	running_val_loss = 0

	# Train
	unet.train()
	logging.debug("Training ...")
	for i, batch_of_patches in tqdm(enumerate(train_patch_loader)):

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

		running_train_loss += train_loss.item()
		break
	running_train_loss /= len(batch_of_patches)

	# Clear CUDA cache
	torch.cuda.empty_cache()


	# Validate
	logging.debug("Validating ...")
	unet.eval() # Set the model in inference mode
	for i, batch_of_patches in tqdm(enumerate(val_patch_loader)):

		PET_patches = batch_of_patches['PET'].cuda()
		GTV_labelmap_patches = batch_of_patches['GTV-labelmap'].long().cuda()

		# Forward pass
		optimizer.zero_grad()
		pred_patches = unet(PET_patches)

		# Compute validation loss
		with torch.no_grad(): # Disable autograd
			val_loss = criterion(pred_patches, GTV_labelmap_patches)
			running_val_loss += val_loss.item()
		break
	running_val_loss /= len(batch_of_patches)

	# Clear CUDA cache
	torch.cuda.empty_cache()

	logging.debug(f"Avg. training loss: {running_train_loss}")
	logging.debug(f"Avg. validation loss: {running_val_loss}")
