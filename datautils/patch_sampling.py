"""
The word "subject" used here doesn't mean any association with the TorchIO Subject class.
The name is just convenient and hence is used here to define a set of volumes belonging to a single patient.
"""

import random
from itertools import islice
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PatchSampler3D():
    """
    Samples 3D patches of specified size using the specified sampling method.
    """
    def __init__(self, patch_size, sampling='random'):
        self.patch_size = list(patch_size) # Specified in (W,H,D) order
        self.patch_size.reverse()    # Convert to (D,H,W) order
        self.sampling = sampling # TODO - This is unused for now. incorporate it in _sample_valid_focal_points() function


    def get_samples(self, subject_dict, num_patches):
        # Sample valid focal points
        focal_points = self._sample_valid_focal_points(subject_dict, num_patches)

        # Get patches from the subject volumes
        patches_list = []  # List of dicts
        for f_pt in focal_points:
            patch = {}
            f_pt = np.array(f_pt)
            start_idx = (f_pt - np.array(self.patch_size)//2).astype(np.int)
            end_idx = (f_pt + np.array(self.patch_size)//2).astype(np.int)

            for key in subject_dict.keys():
                if key == 'PET' or key == 'CT': # The shape is (C,D,H,W) for PET and CT.
                    patch[key] = subject_dict[key][:, start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
                else:  # Labelmap has shape (D,H,W)
                    patch[key] = subject_dict[key][start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]

            patches_list.append(patch)

        return patches_list


    def _sample_valid_focal_points(self, subject_dict, num_patches):
        # Use the labelmap to determine volume shape
        volume_shape = subject_dict['GTV-labelmap'].shape # (D,H,W)
        patch_size = np.array(self.patch_size).astype(np.float)
        valid_indx_range = [
                            np.zeros(3) + np.ceil(patch_size/2),
                            np.array(volume_shape) - np.floor(patch_size/2 - 1)
                            ]
        zs = np.random.randint(valid_indx_range[0][0], valid_indx_range[1][0], num_patches)
        ys = np.random.randint(valid_indx_range[0][1], valid_indx_range[1][1], num_patches)
        xs = np.random.randint(valid_indx_range[0][2], valid_indx_range[1][2], num_patches)
        focal_points = [(zs[i], ys[i], xs[i]) for i in range(num_patches)]
        return focal_points


class PatchSampler2D():
    """
    Samples 2D axial slice patches of specified x-y size using the specified sampling method.
    """
    def __init__(self, patch_size, sampling='random'):
        self.patch_size = list(patch_size) # Specified in (W,H) order
        self.patch_size.reverse()    # Convert to (H,W) order
        self.sampling = sampling # TODO - This is unused for now. incorporate it in _sample_valid_focal_points() function


    def get_samples(self, subject_dict, num_patches):
        # Sample valid focal points
        focal_points = self._sample_valid_focal_points(subject_dict, num_patches)

        # Get patches from the subject volumes
        slice_patches_list = []  # List of dicts
        for f_pt in focal_points:
            slice_patch = {}
            f_pt = np.array(f_pt)
            z = f_pt[0]
            xy_start_idx = (f_pt[1:] - np.array(self.patch_size)//2).astype(np.int)
            xy_end_idx = (f_pt[1:] + np.array(self.patch_size)//2).astype(np.int)

            for key in subject_dict.keys():
                if key == 'PET' or key == 'CT': # The shape is (C,D,H,W) for PET and CT.
                    slice_patch[key] = subject_dict[key][:, z, xy_start_idx[0]:xy_end_idx[0], xy_start_idx[1]:xy_end_idx[1]]
                else:  # Labelmap has shape (D,H,W)
                    slice_patch[key] = subject_dict[key][z, xy_start_idx[0]:xy_end_idx[0], xy_start_idx[1]:xy_end_idx[1]]

            slice_patches_list.append(slice_patch)

        return slice_patches_list


    def _sample_valid_focal_points(self, subject_dict, num_patches):
        # Use the labelmap to determine volume shape
        volume_shape = subject_dict['GTV-labelmap'].shape # (D,H,W)
        patch_size = np.array(self.patch_size).astype(np.float)
        valid_indx_range = [
                            np.zeros(2) + np.ceil(patch_size/2),
                            np.array(volume_shape[1:]) - np.floor(patch_size/2 - 1)
                            ]
        zs = np.random.randint(0, volume_shape[0]-1, num_patches)
        ys = np.random.randint(valid_indx_range[0][0], valid_indx_range[1][0], num_patches)
        xs = np.random.randint(valid_indx_range[0][1], valid_indx_range[1][1], num_patches)
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

        # Shuffle the queue
        if self.shuffle_patches:
            random.shuffle(self.patches_list)

    def _get_next_subject_sample(self):
        # A StopIteration exception is expected when the queue is empty
        try:
            subject_sample = next(self.subjects_iterable)
        except StopIteration as exception:
            self._print('Queue is empty:', exception)
            self.subjects_iterable = self._get_subjects_iterable()
            subject_sample = next(self.subjects_iterable)
        return subject_sample

    def _get_subjects_iterable(self):
        subjects_loader = DataLoader(self.dataset,
                                     num_workers=self.num_workers,
                                     collate_fn=lambda x: x[0],
                                     shuffle=self.shuffle_subjects,
                                    )
        return iter(subjects_loader)


if __name__ == '__main__':

    import sys
    sys.path.append("../")
    from datasets.HECKTORUnimodalityDataset import HECKTORUnimodalityDataset
    from datautils.preprocessing import Preprocessor
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    data_dir = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
    patient_id_filepath = "../hecktor_meta/patient_IDs_train.txt"
    preprocessor = Preprocessor()
    PET_dataset = HECKTORUnimodalityDataset(data_dir,
                                    patient_id_filepath,
                                    mode='training',
                                    preprocessor=preprocessor,
                                    input_modality='PET',
                                    augment_data=False)

    patch_sampler = PatchSampler2D(patch_size=(450,450))
    patch_queue = PatchQueue(PET_dataset,
                         max_length=16,
                         samples_per_volume=8,
                         sampler=patch_sampler,
                         num_workers=0,
                         shuffle_subjects=True,
                         shuffle_patches=True)

    patch_loader = DataLoader(patch_queue, batch_size=4)

    fig, axs = plt.subplots(1,2)
    for i, batch_of_patches in enumerate(patch_loader):
        print("Batch:",i+1)
        print(batch_of_patches['PET'].shape)
        PET_np = batch_of_patches['PET'][0,0,:,:].numpy()
        PET_np = np.clip(PET_np, -2, 8)
        axs[0].imshow(PET_np, cmap='gist_rainbow')
        axs[1].imshow(batch_of_patches['GTV-labelmap'][0,:,:])
        plt.show()
        break