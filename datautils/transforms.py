import numpy as np
import torchio

# TorchIO augmentation config
ROTATION_RANGE = (-10, 10)
SCALE_FACTOR_RANGE = (0.85,1.15)
NUM_CONTROL_POINTS = (5,5,5)
MAX_DISPLACEMENT = (20,20,20)

# PET intensity stretching
PET_PC95_INCREASE_FACTOR = 1.2



def build_transforms():

		# TorchIO transforms
		rotation_transform = torchio.RandomAffine(scales=(1,1), degrees=ROTATION_RANGE, translation=(0,0))
		scaling_transform = torchio.RandomAffine(scales=SCALE_FACTOR_RANGE, degrees=(0,0), translation=(0,0))
		elastic_transform = torchio.RandomElasticDeformation(num_control_points=NUM_CONTROL_POINTS, max_displacement=MAX_DISPLACEMENT, locked_borders=2)
		transforms_dict = {rotation_transform: 0.33,
		                    scaling_transform: 0.33,
		                    elastic_transform: 0.33}
		torchio_oneof_transform = torchio.transforms.OneOf(transforms_dict)


		# Intensity stretching for PET
		def PET_stretch_transform(PET_np):
			# Sstretch the contrast in the range between 30 percentile and 95 percentile
			pc30 = np.percentile(PET_np, 30)
			pc95 = np.percentile(PET_np, 95)
			max_suv = PET_np.max()
			PET_stretched_np = PET_np.copy()

			# Stretch the contrast in range [pc30, pc95)
			mask = (PET_np >= pc30) & (PET_np < pc95)
			PET_stretched_np[mask] = (PET_np[mask]-pc30)/(pc95-pc30) * (PET_PC95_INCREASE_FACTOR*pc95-pc30) + pc30

			# Squeeze the contrast in range [pc95, max]
			mask = (PET_np >= pc95) & (PET_np <= max_suv)
			PET_stretched_np[mask] = (PET_np[mask]-pc95)/(max_suv-pc95) * (max_suv-PET_PC95_INCREASE_FACTOR*pc95) + PET_PC95_INCREASE_FACTOR*pc95

			return PET_stretched_np


		return torchio_oneof_transform, PET_stretch_transform