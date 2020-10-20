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
        self.sampling = sampling # TODO - Add new schemes


    def get_samples(self, subject_dict, num_patches):
        # Sample valid focal points
        focal_points = self._sample_valid_focal_points(subject_dict, num_patches)

        # Extract patches from the subject volumes
        patch_size = np.array(self.patch_size)
        patches_list = []  # List of dicts
        for f_pt in focal_points:
            patch = {}
            f_pt = np.array(f_pt)
            start_idx = (f_pt - np.floor(patch_size/2)).astype(np.int)
            end_idx = (f_pt + np.ceil(patch_size/2)).astype(np.int)
            # print(f_pt, start_idx, end_idx)

            for key in subject_dict.keys():
                if key == 'PET' or key == 'CT' or key == 'PET-CT': # The shape is (C,D,H,W) for PET and CT.
                    patch[key] = subject_dict[key][:, start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
                else:  # Labelmap has shape (D,H,W)
                    patch[key] = subject_dict[key][start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]

            patches_list.append(patch)

        return patches_list


    def _sample_valid_focal_points(self, subject_dict, num_patches):
        # Use the labelmap to determine volume shape
        volume_shape = subject_dict['GTV-labelmap'].shape # (D,H,W)
        patch_size = np.array(self.patch_size).astype(np.float)

        # Get valid index range for focal points - upper-bound inclusive
        valid_indx_range = [
                            np.zeros(3) + np.floor(patch_size/2),
                            np.array(volume_shape) - np.ceil(patch_size/2)
                            ]

        if self.sampling == 'random':
            # randint takes inclusive range
            zs = np.random.randint(valid_indx_range[0][0], valid_indx_range[1][0], num_patches)
            ys = np.random.randint(valid_indx_range[0][1], valid_indx_range[1][1], num_patches)
            xs = np.random.randint(valid_indx_range[0][2], valid_indx_range[1][2], num_patches)
        elif self.sampling == 'sequential':
            # arange takes exlusive range
            z_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1).astype(np.int)
            y_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1).astype(np.int)
            x_range = np.arange(valid_indx_range[0][2], valid_indx_range[1][2] + 1).astype(np.int)
            zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
            zs, ys, xs = zs.flatten(), ys.flatten(), xs.flatten()

        focal_points = [(zs[i], ys[i], xs[i]) for i in range(num_patches)]
        return focal_points


class PatchSampler2D():
    """
    Samples 2D axial slice patches of specified x-y size using the specified sampling method.
    """
    def __init__(self, patch_size, sampling='random'):
        self.patch_size = list(patch_size) # Specified in (W,H) order
        self.patch_size.reverse()    # Convert to (H,W) order
        self.sampling = sampling # TODO


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
        volume_shape = subject_dict['GTV-labelmap'].shape # (D,H,W)
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
            y_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1).astype(np.int)
            x_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1).astype(np.int)
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




def get_num_valid_patches(patch_size, volume_size=(450,450,100)):
    dimensions = len(patch_size)
    num_slices = volume_size[2]
    if dimensions == 2 and len(volume_size) == 3:
        volume_size = volume_size[:2]
    patch_size = np.array(patch_size)

    # Get inclusive range of valid focal points
    valid_indx_range = [
                        np.zeros(dimensions) + np.floor(patch_size/2),
                        np.array(volume_size) - np.ceil(patch_size/2)
                       ]

    valid_area_size = valid_indx_range[1] - valid_indx_range[0] + 1
    if dimensions == 2:   num_valid_focal_pts = valid_area_size[0] * valid_area_size[1] * num_slices
    elif dimensions == 3:   num_valid_focal_pts = valid_area_size[0] * valid_area_size[1] * valid_area_size[2]
    return int(num_valid_focal_pts)





if __name__ == '__main__':

    import sys
    sys.path.append("../")
    from datasets.hecktor_petct_dataset import HECKTORPETCTDataset
    from datasets.hecktor_unimodal_dataset import HECKTORUnimodalDataset
    from datautils.preprocessing import Preprocessor
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    data_dir = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
    patient_id_filepath = "../hecktor_meta/patient_IDs_train.txt"
    preprocessor = Preprocessor()



    def test_patch_orientation():
        PET_CT_dataset = HECKTORPETCTDataset(data_dir,
                                    patient_id_filepath,
                                    mode='training',
                                    preprocessor=preprocessor,
                                    input_representation='separate-volumes',
                                    augment_data=False)

        patch_sampler = PatchSampler2D(patch_size=(450,450))
        patch_queue = PatchQueue(PET_CT_dataset,
                             max_length=16,
                             samples_per_volume=8,
                             sampler=patch_sampler,
                             num_workers=0,
                             shuffle_subjects=True,
                             shuffle_patches=True)

        patch_loader = DataLoader(patch_queue, batch_size=1)

        fig, axs = plt.subplots(1,3)
        for i, batch_of_patches in enumerate(patch_loader):
            print("Batch:",i+1)
            print(batch_of_patches['PET'].shape)
            PET_np = batch_of_patches['PET'][0,0,:,:].numpy()
            CT_np = batch_of_patches['CT'][0,0,:,:].numpy()
            axs[0].imshow(np.clip(PET_np, -2, 8), cmap='gist_rainbow')
            axs[1].imshow(np.clip(CT_np, -150, 150), cmap='gray')
            axs[2].imshow(batch_of_patches['GTV-labelmap'][0,:,:], cmap='gray')
            plt.show()
            break


    def test_dataloader_limit():
        PET_dataset = HECKTORUnimodalDataset(data_dir,
                                    patient_id_filepath,
                                    mode='cval-CHMR-validation',
                                    preprocessor=preprocessor,
                                    input_modality='PET',
                                    augment_data=False)
        print("Dataset length:", len(PET_dataset))

        patch_sampler = PatchSampler2D(patch_size=(128,128))
        patch_queue = PatchQueue(PET_dataset,
                             max_length=4,
                             samples_per_volume=1,
                             sampler=patch_sampler,
                             num_workers=0,
                             shuffle_subjects=True,
                             shuffle_patches=True)

        print("patch queue length:", len(patch_queue))

        patch_loader = DataLoader(patch_queue, batch_size=1)

        print("Testing patch loader length ...")
        counter = 0
        for batch_of_patchs in patch_loader:
            counter += 1
            print(counter)


    def test_sequential_sampling():
        PET_CT_dataset = HECKTORPETCTDataset(data_dir,
                                    patient_id_filepath,
                                    mode='cval-CHMR-validation',
                                    preprocessor=preprocessor,
                                    input_representation='multichannel-volume',
                                    augment_data=False)
        subject_dict = PET_CT_dataset[0]

        print("Testing 3D ...")
        patch_size_3d = (448,448,98)
        print("Total valid patches:", get_num_valid_patches(patch_size_3d))
        patch_sampler_3d = PatchSampler3D(patch_size=patch_size_3d, sampling='sequential')
        focal_pts = patch_sampler_3d._sample_valid_focal_points(subject_dict, get_num_valid_patches(patch_size_3d))
        print("Focal pts:", focal_pts)
        sample_patch = patch_sampler_3d.get_samples(subject_dict, num_patches=1)[0]
        print("Sample patch shape:", sample_patch['PET-CT'].shape, sample_patch['GTV-labelmap'].shape)

        print("\nTesting 2D ...")
        patch_size_2d = (448,448)
        print("Total valid patches:", get_num_valid_patches(patch_size_2d))
        patch_sampler_2d = PatchSampler2D(patch_size=patch_size_2d, sampling='sequential')
        focal_pts = patch_sampler_2d._sample_valid_focal_points(subject_dict, get_num_valid_patches(patch_size_2d))
        #print("Focal pts:", focal_pts)
        sample_patch = patch_sampler_2d.get_samples(subject_dict, num_patches=1)[0]
        print("Sample patch shape:", sample_patch['PET-CT'].shape, sample_patch['GTV-labelmap'].shape)



    test_sequential_sampling()