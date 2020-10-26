import numpy as np
import torch


class PatchAggregator3D():
    def __init__(self, patch_size=[128,128,32], volume_size=[144,144,48], focal_point_stride=[5,5,5], overlap_handling=None, unpadding=[0,0,0]):

        """
        All arguments are apecified in (W,H,D) format
        """

        self.patch_size = list(patch_size)
        self.volume_size = list(volume_size)
        self.focal_point_stride = list(focal_point_stride)

        # Convert to (D,H,W) ordering
        self.patch_size.reverse()
        self.volume_size.reverse()
        self.focal_point_stride.reverse()

        self.overlap_handling = overlap_handling  # None or 'union'

        self.unpadding = list(unpadding)
        self.unpadding.reverse()

        # Increase the volume size to account for the padding used during patch sampling
        self.volume_size = [self.volume_size[i] + self.unpadding[i] for i in range(3)] # (D,H,W)

        self.valid_focal_points = self._get_valid_focal_points() # Valid focal points in volume coordinates


    def _get_valid_focal_points(self):
        patch_size = np.array(self.patch_size)
        valid_indx_range = [
                            np.zeros(3) + np.floor(patch_size/2),
                            np.array(self.volume_size) - np.ceil(patch_size/2)
                           ]

        z_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1, self.focal_point_stride[0]).astype(np.int)
        y_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1, self.focal_point_stride[1]).astype(np.int)
        x_range = np.arange(valid_indx_range[0][2], valid_indx_range[1][2] + 1, self.focal_point_stride[2]).astype(np.int)
        zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
        zs, ys, xs = zs.flatten(), ys.flatten(), xs.flatten()

        valid_focal_points = [(zs[i], ys[i], xs[i]) for i in range(zs.shape[0])]
        return valid_focal_points


    def aggregate(self, patches_list, device='cpu'):
        """
        Args:
            patches_list: List of torch tensors arrays. Shape of the each array is in (D,H,W) ordering
        Returns:
            full_volume: Tensor. Aggregated from the patches. Shape has dim (D,H,W) ordering
        """
        # Define a zeros array of shape volume_size
        full_volume = torch.zeros(self.volume_size, device=device)

        patch_size = np.array(self.patch_size)

        for i, patch in enumerate(patches_list):

            # Find the indices of the volume where the patch needs to be placed
            global_focal_point = np.array(self.valid_focal_points[i])
            global_start_idxs = global_focal_point.astype(np.int) - np.floor(patch_size/2).astype(np.int)
            z1, y1, x1 = global_start_idxs
            z2, y2, x2 = global_start_idxs + patch_size

            # Handle overlap or not
            if self.overlap_handling is None:
                full_volume[z1:z2, y1:y2, x1:x2] = patch

            elif self.overlap_handling == 'union':
                full_volume_copy = full_volume.clone().detach()
                full_volume[z1:z2, y1:y2, x1:x2] = patch.clone().detach()
                full_volume = torch.max(full_volume, full_volume_copy)

        # If padding was used during patch sampling, remove it from the full volume
        if self.unpadding != [0,0,0]:
            full_volume = full_volume[:-self.unpadding[0], :-self.unpadding[1], :-self.unpadding[2]].clone().detach()

        return full_volume



def get_pred_labelmap_patches_list(pred_prob_patches):
    """
    Get a list of predicted labelmap patches from the the model's predicted batch of probabilities patches

    Args:
        pred_prob_patches: Tensor. Batch of predicted probabilites patches. Shape (N,C,D,H,W)
    Returns:
        pred_labelmap_patches_list: List (length N) of tensors. Each element is a tensor of shape (D,H,W)
    """
    pred_labelmap_patches_list = []

    for i in range(pred_prob_patches.shape[0]):

        # Convert to numpy, collapse channel dim for the predicted patch
        pred_patch = pred_prob_patches[i] # Shape (C,D,H,W)
        pred_patch = pred_patch.argmax(dim=0)  # Shape (D,H,W)

        # Accumulate in the list
        pred_labelmap_patches_list.append(pred_patch)

    return pred_labelmap_patches_list




def get_pred_labelmap_patches_list(pred_prob_patches):
    """
    Get a list of predicted labelmap patches from the the model's predicted batch of probabilities patches

    Args:
        pred_prob_patches: Tensor. Batch of predicted probabilites patches. Shape (N,C,D,H,W)
    Returns:
        pred_labelmap_patches_list: List of length N. Each element is a tensor of shape (D,H,W)
    """
    pred_labelmap_patches_list = []

    for i in range(pred_prob_patches.shape[0]):

        # Convert to numpy, collapse channel dim for the predicted patch
        pred_patch = pred_prob_patches[i] # Shape (C,D,H,W)
        pred_patch = pred_patch.argmax(dim=0)  # Shape (D,H,W)

        # Accumulate in the list
        pred_labelmap_patches_list.append(pred_patch)

    return pred_labelmap_patches_list

