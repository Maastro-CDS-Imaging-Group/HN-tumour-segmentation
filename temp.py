import torch
from datautils.patch_sampling import PatchSampler3D, get_num_valid_patches



patch_size = [144,144,48]
volume_size = [450,450,100]
focal_point_stride = [140,140,44]
padding = [114,114,36]

num_valid_patches = get_num_valid_patches(patch_size, volume_size, focal_point_stride, padding)

print(num_valid_patches)


sampler = PatchSampler3D(patch_size, volume_size, "sequential", focal_point_stride, padding)
focal_points = sampler._sample_valid_focal_points(num_patches=num_valid_patches)