import numpy as np
from scipy.ndimage import gaussian_filter


# Histogram related constants
# Standard range for PET is set to [0, 20] SUV and for CT is set to [-250,150] -- same as their clipping ranges
DEFAULT_QUANTILES_CUTOFF_SUV = (0, 1)
DEFAULT_QUANTILES_CUTOFF_HU = (0, 1)


class Preprocessor():

    def __init__(self,
                 smooth_sigma_mm={'PET': 2.0, 'CT': 0.0},
                 standardization_method={'PET': 'clipping', 'CT': 'clipping'},
                 clipping_range={'PET': [0,20], 'CT': [-150,150]},
                 histogram_landmarks_path={'PET': None, 'CT': None}
                 ):

        # Smoothing params
        self.spacing_dict = None
        self.smooth_sigma_mm = smooth_sigma_mm

        # Standardization params
        self.standardization_method = standardization_method
        self.clipping_range = clipping_range
        self.histogram_landmarks_path = histogram_landmarks_path
        self.landmarks_dict = {'PET': None, 'CT': None}
        self.quantiles_cutoff = {'PET': DEFAULT_QUANTILES_CUTOFF_SUV, 'CT': DEFAULT_QUANTILES_CUTOFF_HU}
        for key in ['PET', 'CT']:
            if standardization_method[key] == 'clipping+histogram':
                self.landmarks_dict[key] = np.load(histogram_landmarks_path[key])


    def set_spacing(self, spacing_dict):
        self.spacing_dict = spacing_dict


    def smoothing_filter(self, image_np, modality):
        sigma_mm = self.smooth_sigma_mm[modality]
        sigma = (sigma_mm / self.spacing_dict['xy spacing'],
                 sigma_mm / self.spacing_dict['xy spacing'],
                 sigma_mm / self.spacing_dict['slice thickness'])
        image_np = gaussian_filter(image_np, sigma=sigma)
        return image_np


    def standardize_intensity(self, image_np, modality):
        if self.standardization_method[modality] == 'clipping':
            image_np = self._clip(image_np, modality)
        elif self.standardization_method[modality] == 'clipping+histogram':
            # Clip the image first to keep intensities in the required range, then apply histogram transform
            image_np = self._clip(image_np, modality)
            image_np = self._histogram_transform(image_np, modality)
        return image_np

    def _clip(self, image_np, modality):
        clipping_range = self.clipping_range[modality]
        image_np = np.clip(image_np, clipping_range[0], clipping_range[1])
        return image_np

    def _histogram_transform(self, image_np, modality, epsilon=1e-5):
        quantiles_cutoff = self.quantiles_cutoff[modality]
        mapping = self.landmarks_dict[modality]
        shape = image_np.shape
        image_np = image_np.reshape(-1).astype(np.float32)

        range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

        percentiles_cutoff = 100 * np.array(quantiles_cutoff)
        percentiles = self._get_percentiles(percentiles_cutoff)
        percentile_values = np.percentile(image_np, percentiles)

        # Apply linear histogram standardization
        range_mapping = mapping[range_to_use]
        range_perc = percentile_values[range_to_use]
        diff_mapping = np.diff(range_mapping)
        diff_perc = np.diff(range_perc)

        # Handling the case where two landmarks are the same
        # for a given input image. This usually happens when
        # image background is not removed from the image.
        diff_perc[diff_perc < epsilon] = np.inf

        affine_map = np.zeros([2, len(range_to_use) - 1])

        # Compute slopes of the linear models
        affine_map[0] = diff_mapping / diff_perc

        # Compute intercepts of the linear models
        affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

        bin_id = np.digitize(image_np, range_perc[1:-1], right=False)
        lin_img = affine_map[0, bin_id]
        aff_img = affine_map[1, bin_id]
        new_img = lin_img * image_np + aff_img
        new_img = new_img.reshape(shape)
        new_img = new_img.astype(np.float32)
        return new_img

    def _get_percentiles(self, percentiles_cutoff):
        quartiles = np.arange(25, 100, 25).tolist()
        deciles = np.arange(10, 100, 10).tolist()
        all_percentiles = list(percentiles_cutoff) + quartiles + deciles
        percentiles = sorted(set(all_percentiles))
        return np.array(percentiles)


    def rescale_to_unit_range(self, image_np):
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        return image_np


if __name__ == '__main__':

    pass