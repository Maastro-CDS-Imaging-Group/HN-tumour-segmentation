import numpy as np
import torch
import SimpleITK as sitk



# Between SimpleITK Image and Numpy ndarray
def sitk2np(image_sitk, keep_whd_ordering=True):
	image_np = sitk.GetArrayFromImage(image_sitk)
	if keep_whd_ordering:
		image_np = image_np.transpose((2,1,0))
	return image_np


def np2sitk(image_np, has_whd_ordering=True, spacing=(1,1,3)):
	if has_whd_ordering:
		image_np = image_np.transpose((2,1,0)) # Convert to (D,H,W)
	image_sitk = sitk.GetImageFromArray(image_np)
	image_sitk.SetSpacing(spacing)
	return image_sitk



# Between Torch tensor and Numpy ndarray
def tensor2np(image_tensor):
	image_np = image_tensor.cpu().numpy()
	return image_np


def np2tensor(image_np):
	image_tensor = torch.from_numpy(image_np)
	return image_tensor



# Between SimpleITK Image and Torch tensor
def sitk2tensor(image_sitk, keep_whd_ordering=True):
	image_np = sitk.GetArrayFromImage(image_sitk)
	if keep_whd_ordering:
		image_np = image_np.transpose((2,1,0))
	image_tensor = torch.from_numpy(image_np)
	return image_tensor


def tensor2sitk(image_tensor, has_whd_ordering=True, spacing=(1,1,3)):
	image_np = image_tensor.cpu().numpy()
	if has_whd_ordering:
		image_np = image_np.transpose((2,1,0)) # Convert to (D,H,W)
	image_sitk = sitk.GetImageFromArray(image_np)
	image_sitk.SetSpacing(spacing)
	return image_sitk


# Dim ordering conversion -- both functions do the exact same thing, but their names make things clear
def whd2dhw(image):
	if isinstance(image, np.ndarray):
		 return image.transpose((2,1,0))
	if isinstance(image, torch.Tensor):
		 return image.permute(2,1,0)

def dhw2whd(image):
	if isinstance(image, np.ndarray):
		 return image.transpose((2,1,0))
	if isinstance(image, torch.Tensor):
		 return image.permute(2,1,0)
