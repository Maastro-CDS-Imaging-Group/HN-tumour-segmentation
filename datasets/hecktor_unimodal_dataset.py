import sys, random

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torchio
import SimpleITK as sitk

sys.path.append("../")
from datautils.conversion import *
import datautils.transforms as transforms


# Constants
AUG_PROBABILITY = 0.5


class HECKTORUnimodalDataset(torch.utils.data.Dataset):
	"""
	Dataset class to interface with the HECKTOR data. Depending on the input_modality parameter, either PET or CT will be loaded.
	The GTV masks for only the HECKTOR train set (4 centres from Quebec) are available.
	Centres and patient distribution:
		- CHGJ -- 55
		- CHMR -- 18
		- CHUM -- 72
		- CHUS -- 56
	"""
	def __init__(self, data_dir, patient_id_filepath, mode='training', preprocessor=None, input_modality='PET', augment_data=False):
		"""
		Parameters:
			data_dir
			patient_id_filepath
			mode -- For default split: 'training', 'validation' -- (Takes CHUM for validation)
					  For cross validation: 'cval-CHGJ-training', 'cval-CHGJ-validation', ...
			input_modality -- 'PET' or 'CT'
			augment_data -- True or False
		"""
		self.data_dir = data_dir
		with open(patient_id_filepath, 'r') as pf:
			self.patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

		self.mode = mode

		# Default train-val split -- CHGJ, CHMR and CHUS for training. CHUM for validation
		if self.mode == 'training':
			self.patient_ids = [p_id for p_id in self.patient_ids if 'CHUM' not in p_id]
		elif self.mode == 'validation':
			self.patient_ids = [p_id for p_id in self.patient_ids if 'CHUM' in p_id]

		# Cross validation option
		if 'cval' in self.mode:
			val_centre = self.mode.split('-')[1]
			if 'training' in self.mode:
				self.patient_ids = [p_id for p_id in self.patient_ids if val_centre not in p_id]
			elif 'validation' in self.mode:
				self.patient_ids = [p_id for p_id in self.patient_ids if val_centre in p_id]

		self.input_modality = input_modality

		self.spacing_dict = {'xy spacing': 1.0, 'slice thickness': 3.0}
		self.preprocessor = preprocessor
		if self.preprocessor is None:
			raise Exception("Specify the preprocessor")
		preprocessor.set_spacing(self.spacing_dict)

		# Augmentation config
		self.augment_data = augment_data
		self.affine_matrix = np.array(      # Affine matrix representing the (1,1,3) spacing as scaling
		                              [
		                               [1,0,0,0],
                                       [0,1,0,0],
                  	                   [0,0,3,0],
                                       [0,0,0,1]
                                      ]
                                     )
		self.torchio_oneof_transform = None
		self.PET_stretch_transform = None
		if self.augment_data:
			self.torchio_oneof_transform, self.PET_stretch_transform = transforms.build_transforms()


	def __len__(self):
		return len(self.patient_ids)


	def __getitem__(self, idx):
		p_id = self.patient_ids[idx]

		# Read data files into sitk images -- (W,H,D) format
		if self.input_modality == 'PET':
			input_image_sitk = sitk.ReadImage(f"{self.data_dir}/{p_id}_pt.nii.gz")
		elif self.input_modality == 'CT':
			input_image_sitk = sitk.ReadImage(f"{self.data_dir}/{p_id}_ct.nii.gz")
		GTV_labelmap_sitk = sitk.ReadImage(f"{self.data_dir}/{p_id}_ct_gtvt.nii.gz")

		# Convert to ndarrays -- Keep the (W,H,D) dim ordering
		input_image_np = sitk2np(input_image_sitk, keep_whd_ordering=True)
		GTV_labelmap_np = sitk2np(GTV_labelmap_sitk, keep_whd_ordering=True)

		# Smooth PET and CT
		input_image_np = self.preprocessor.smoothing_filter(input_image_np, modality=self.input_modality)

		# Standardize the intensity scale
		input_image_np = self.preprocessor.standardize_intensity(input_image_np, modality=self.input_modality)

		# Data augmentation
		if self.augment_data:
			if random.random() < AUG_PROBABILITY:
				input_image_np, GTV_labelmap_np = self.apply_transform(input_image_np, GTV_labelmap_np)

		# Rescale intensities to [0,1] range
		# input_image_np = self.preprocessor.rescale_to_unit_range(input_image_np)

		# Construct the sample dict -- Convert to tensor and change dim ordering to (D,H,W).
		# Input image will have shape (1,D,H,W). Target labelmap will have (D,H,W)
		sample_dict = {self.input_modality: np2tensor(input_image_np).permute(2,1,0).unsqueeze(dim=0),
                       'GTV-labelmap': np2tensor(GTV_labelmap_np).permute(2,1,0)
		              }

		return sample_dict


	def apply_transform(self, input_image_np, GTV_labelmap_np):
		r = random.random()
		if  r < 0.75:
			# Apply one of the 3 TorchIO spatial transforms. Need to pack the volumes into a TorchIO Subject for this.
			subject_tio = self._create_torchio_subject(input_image_np, GTV_labelmap_np)
			subject_tio = self.torchio_oneof_transform(subject_tio)
			input_image_np = subject_tio[self.input_modality].numpy().squeeze()
			GTV_labelmap_np = subject_tio['GTV-labelmap'].numpy().squeeze()
		else:
			# PET intensity stretching
			if self.input_modality == 'PET':
				input_image_np = self.PET_stretch_transform(input_image_np)
		return input_image_np, GTV_labelmap_np

	def _create_torchio_subject(self, input_image_np, GTV_labelmap_np):
		input_image_tio = torchio.Image(tensor=np2tensor(input_image_np).unsqueeze(dim=0), type=torchio.INTENSITY, affine=self.affine_matrix)
		GTV_labelmap_tio = torchio.Image(tensor=np2tensor(GTV_labelmap_np).unsqueeze(dim=0), type=torchio.LABEL, affine=self.affine_matrix)
		subject_dict = {self.input_modality: input_image_tio, 'GTV-labelmap': GTV_labelmap_tio}
		subject_tio = torchio.Subject(subject_dict)
		return subject_tio



if __name__ == '__main__':

	from data_utils.preprocessing import Preprocessor

	data_dir = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
	patient_id_filepath = "../hecktor_meta/patient_IDs_train.txt"
	preprocessor = Preprocessor()

	dataset = HECKTORUnimodalityDataset(data_dir,
			                          patient_id_filepath,
			                          mode='training',
			                          preprocessor=preprocessor,
			                          input_modality='PET',
			                          augment_data=False)

	sample = dataset[0]
	print(sample['PET'].shape)