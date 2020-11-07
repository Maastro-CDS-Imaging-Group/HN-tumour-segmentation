"""
The word "subject" used here doesn't mean any association with the TorchIO Subject class.
The name is just convenient and hence is used here to define a set of volumes belonging to a single patient.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PatchSampler3D():
    """
    Samples 3D patches of specified size using the specified sampling method.
    """
    def __init__(self, patch_size, volume_size, sampling, focal_point_stride, padding):
        self.patch_size = list(patch_size) # Specified in (W,H,D) order
        self.patch_size.reverse()    # Convert to (D,H,W) order

        self.volume_size = list(volume_size) # Specified in (W,H,D) order
        self.volume_size.reverse()  # Convert to (D,H,W) order

        self.sampling = sampling

        self.focal_point_stride = list(focal_point_stride) # Useful while using sequential sampling. Specified in (W,H,D) order
        self.focal_point_stride.reverse() # Convert to (D,H,W) order

        self.padding = list(padding) # One-sided zero padding, along each dim.
        self.padding.reverse() # Convert to (D,H,W) order
        if self.padding != [0,0,0]:
            self.volume_size = [self.volume_size[i] + self.padding[i] for i in range(3)]


    def get_samples(self, subject_dict, num_patches):
        # Sample valid focal points
        focal_points = self._sample_valid_focal_points(num_patches)

        # Extract patches from the subject volumes
        patch_size = np.array(self.patch_size)
        patches_list = []  # List of dicts
        for f_pt in focal_points:
            patch = {}
            f_pt = np.array(f_pt)
            start_idx = f_pt.astype(np.int) - np.floor(patch_size/2).astype(np.int)
            z1, y1, x1 = start_idx
            end_idx = start_idx.astype(np.int) + patch_size.astype(np.int)
            z2, y2, x2 = end_idx

            for key in subject_dict.keys():
                volume = subject_dict[key].numpy()

                if key == 'PET' or key == 'CT' or key == 'PET-CT': # The shape is (C,D,H,W) for PET and CT.
                    if self.padding != [0,0,0]:    volume = np.pad(volume, pad_width=[(0,0), (0,self.padding[0]), (0,self.padding[1]), (0,self.padding[2])])
                    patch[key] = volume[:, z1:z2, y1:y2, x1:x2]

                elif key == 'target-labelmap':  # Labelmap has shape (D,H,W)
                    if self.padding != [0,0,0]:   volume = np.pad(volume, pad_width=[(0,self.padding[0]), (0,self.padding[1]), (0,self.padding[2])])
                    patch[key] = volume[z1:z2, y1:y2, x1:x2]

                patch[key] = torch.from_numpy(patch[key])
                #subject_dict[key] = torch.from_numpy(subject_dict[key])

            patches_list.append(patch)

        return patches_list


    def _sample_valid_focal_points(self, num_patches):
        # Use the labelmap to determine volume shape
        #volume_shape = subject_dict['target-labelmap'].shape # (D,H,W)
        patch_size = np.array(self.patch_size).astype(np.float)

        # Get valid index range for focal points - upper-bound inclusive
        valid_indx_range = [
                            np.zeros(3) + np.floor(patch_size/2),
                            np.array(self.volume_size) - np.ceil(patch_size/2)
                           ]

        if self.sampling == 'random':
            # Uniform random over all valid focal points
            # Note: randint() takes inclusive range
            zs = np.random.randint(valid_indx_range[0][0], valid_indx_range[1][0], num_patches)
            ys = np.random.randint(valid_indx_range[0][1], valid_indx_range[1][1], num_patches)
            xs = np.random.randint(valid_indx_range[0][2], valid_indx_range[1][2], num_patches)
            focal_points = [(zs[i], ys[i], xs[i]) for i in range(num_patches)]

        elif self.sampling == 'sequential':
            # Sequental sampling, used during inference
            # Note: arange() takes exlusive range
            z_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1, self.focal_point_stride[0]).astype(np.int)
            y_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1, self.focal_point_stride[1]).astype(np.int)
            x_range = np.arange(valid_indx_range[0][2], valid_indx_range[1][2] + 1, self.focal_point_stride[2]).astype(np.int)
            zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
            zs, ys, xs = zs.flatten(), ys.flatten(), xs.flatten()
            focal_points = [(zs[i], ys[i], xs[i]) for i in range(num_patches)]

        elif self.sampling == 'strided-random':
            # Unofrm random sampling over spare valid focal points
            z_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1, self.focal_point_stride[0]).astype(np.int)
            y_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1, self.focal_point_stride[1]).astype(np.int)
            x_range = np.arange(valid_indx_range[0][2], valid_indx_range[1][2] + 1, self.focal_point_stride[2]).astype(np.int)
            zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
            zs, ys, xs = zs.flatten(), ys.flatten(), xs.flatten()
            focal_points = [(zs[i], ys[i], xs[i]) for i in range(num_patches)]
            random_indxs = np.random.choice(len(focal_points), size=num_patches, replace=False)
            focal_points = [focal_points[i] for i in random_indxs]

        elif self.sampling == 'ct-foreground-random':
            # Random sampling from the CT foreground (any value >0)
            # TODO
            pass

        elif self.sampling == 'gtv-biased-random':
            # Random sampling with higher probability for GTV focal points
            # TODO
            pass

        return focal_points


class PatchSampler2D():
    """
    Samples 2D axial slice patches of specified x-y size using the specified sampling method.

    TODO: Old version. Make all the required changes based on PatchSampler3D.

    """
    def __init__(self, patch_size, sampling='random', focal_point_stride=(1,1)):
        self.patch_size = list(patch_size) # Specified in (W,H) order
        self.patch_size.reverse()    # Convert to (H,W) order

        self.sampling = sampling # TODO

        self.focal_point_stride = list(focal_point_stride)  # Useful while using sequential sampling. Specified in (W,H) order
        self.focal_point_stride.reverse()  # Convert to (H,W) order


    def get_samples(self, subject_dict, num_patches):
        # Sample valid focal points
        focal_points = self._sample_valid_focal_points(subject_dict, num_patches)

        # Get patches from the subject volumes
        patch_size = np.array(self.patch_size)
        patches_list = []  # List of dicts
        for f_pt in focal_points:
            patch = {}
            f_pt = np.array(f_pt)
            z = f_pt[0]
            xy_start_idx = (f_pt[1:] - np.floor(patch_size/2)).astype(np.int)
            xy_end_idx = (f_pt[1:] + np.ceil(patch_size/2)).astype(np.int)

            for key in subject_dict.keys():
                if key == 'PET' or key == 'CT' or key == 'PET-CT': # The shape is (C,D,H,W) for PET and CT.
                    patch[key] = subject_dict[key][:, z, xy_start_idx[0]:xy_end_idx[0], xy_start_idx[1]:xy_end_idx[1]]
                else:  # Labelmap has shape (D,H,W)
                    patch[key] = subject_dict[key][z, xy_start_idx[0]:xy_end_idx[0], xy_start_idx[1]:xy_end_idx[1]]

            patches_list.append(patch)

        return patches_list


    def _sample_valid_focal_points(self, subject_dict, num_patches):
        # Use the labelmap to determine volume shape
        volume_shape = subject_dict['target-labelmap'].shape # (D,H,W)
        patch_size = np.array(self.patch_size).astype(np.float)

        # Get valid index range for focal points - upper-bound inclusive
        valid_indx_range = [
                            np.zeros(2) + np.floor(patch_size/2),
                            np.array(volume_shape[1:]) - np.ceil(patch_size/2)
                            ]

        if self.sampling == 'random':
            # randint takes inclusive range
            zs = np.random.randint(0, volume_shape[0]-1, num_patches)
            ys = np.random.randint(valid_indx_range[0][0], valid_indx_range[1][0], num_patches)
            xs = np.random.randint(valid_indx_range[0][1], valid_indx_range[1][1], num_patches)
        elif self.sampling == 'sequential':
            # arange takes exclusive range
            z_range = np.arange(0, volume_shape[0]).astype(np.int)
            y_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1, self.focal_point_stride[0]).astype(np.int)
            x_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1, self.focal_point_stride[1]).astype(np.int)
            zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
            zs, ys, xs = zs.flatten(), ys.flatten(), xs.flatten()

        focal_points = [(zs[i], ys[i], xs[i]) for i in range(num_patches)]
        return focal_points



class PatchQueue(Dataset):

    def __init__(self, dataset, max_length, samples_per_volume, sampler, num_workers, shuffle_subjects=True, shuffle_patches=True):
        self.dataset = dataset
        self.max_length = max_length
        self.samples_per_volume = samples_per_volume
        self.sampler = sampler  # Instance of the custom PatchSampler() class
        self.num_workers = num_workers
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches

        # Additional attributes
        self.total_subjects = len(self.dataset)
        self.iterations_per_epoch = self.total_subjects * self.samples_per_volume
        self.subjects_iterable = self._get_subjects_iterable()

        # Data structures
        self.subject_samples_list = [] # List of dicts -- To store sets of full volumes of fetched patients
        self.patches_list = []  # List of dicts -- Main data structure implementing the patch queue

        self.counter = 0

    def __len__(self):
        return self.iterations_per_epoch


    def __getitem__(self, _):

        if len(self.patches_list) == 0:
            self.fill_queue()

        sample_dict = self.patches_list.pop()
        return sample_dict


    def fill_queue(self):
        # Determine the number of subjects to be read
        max_num_subjects_for_queue = self.max_length // self.samples_per_volume
        num_subjects_for_queue = min(self.total_subjects, max_num_subjects_for_queue)

        # Read the subjects, sample patches from the volumes and populate the queue
        for _ in range(num_subjects_for_queue):
            subject_sample = self._get_next_subject_sample()
            patches = self.sampler.get_samples(subject_sample, self.samples_per_volume)
            self.patches_list.extend(patches)

            self.counter += 1
            # print("[PatchQueue]", self.counter)

        # Shuffle the queue
        if self.shuffle_patches:
            random.shuffle(self.patches_list)

    def _get_next_subject_sample(self):
        # A StopIteration exception is expected when the queue is empty
        try:
            subject_sample = next(self.subjects_iterable)
        except StopIteration as exception:
            # print("[PatchQueue]Subjects loader exhausted. Initializing again")
            self.subjects_iterable = self._get_subjects_iterable()
            subject_sample = next(self.subjects_iterable)
        return subject_sample

    def _get_subjects_iterable(self):
        subjects_loader = DataLoader(self.dataset,
                                     num_workers=self.num_workers,
                                     collate_fn=lambda x: x[0],
                                     shuffle=self.shuffle_subjects,
                                    )
        # print("subjects loader length:", len(subjects_loader))
        return iter(subjects_loader)


def get_num_valid_patches(patch_size, volume_size=[450,450,100], focal_point_stride=[1,1,1], padding=[0,0,0]):

    # Convert to (D,H,W) ordering.   [ (H,W) for patch_size and focal_point_stride if theu are 2D ]
    patch_size = list(patch_size)
    patch_size.reverse()

    volume_size = list(volume_size)
    volume_size.reverse()

    padding = list(padding)
    padding.reverse()

    # Increase the volume size to account for the padding used during patch sampling
    if padding != [0,0,0]:
        volume_size = [volume_size[i] + padding[i] for i in range(3)]

    focal_point_stride = list(focal_point_stride)
    focal_point_stride.reverse()

    assert len(patch_size) == len(focal_point_stride)
    dimensions = len(patch_size)
    num_slices = volume_size[0]
    if dimensions == 2 and len(volume_size) == 3:
        volume_size = volume_size[1:]
    patch_size = np.array(patch_size)

    # Get inclusive range of valid focal points
    valid_indx_range = [
                        np.zeros(dimensions) + np.floor(patch_size/2),
                        np.array(volume_size) - np.ceil(patch_size/2)
                       ]

    valid_region_size = valid_indx_range[1] - valid_indx_range[0] + 1
    if dimensions == 2:
        num_valid_focal_pts = np.ceil(float(valid_region_size[0]/focal_point_stride[0])) * \
                              np.ceil(float(valid_region_size[1]/focal_point_stride[1])) * \
                              np.ceil(float(num_slices))
    elif dimensions == 3:
        num_valid_focal_pts = np.ceil(float(valid_region_size[0]/focal_point_stride[0])) * \
                              np.ceil(float(valid_region_size[1]/focal_point_stride[1])) * \
                              np.ceil(float(valid_region_size[2]/focal_point_stride[2]))

    return int(num_valid_focal_pts)





