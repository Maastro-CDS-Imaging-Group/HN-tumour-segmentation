import os

import numpy as np

import pydicom
import SimpleITK as sitk

def read_dcm_series(series_dir):
    """
    Read a DICOM series into a sitk image.
    Parameters
        series_dir: Directory containing a DICOM series
    Returns
        sitk_image: SimpleITK Image object
        meta: Python dictionary containing selected meta-data
    """
    if len(os.listdir(series_dir)) > 1: # If multiple dcm files in the series (as is the case with CT and PET scans)
        reader = sitk.ImageSeriesReader()
        dcm_file_paths = reader.GetGDCMSeriesFileNames(series_dir)
        sitk_image = sitk.ReadImage(dcm_file_paths)
        dicom_data = pydicom.read_file(dcm_file_paths[0], stop_before_pixels=True)
    else:
        file_name = os.listdir(series_dir)[0]
        sitk_image = None
        dicom_data = pydicom.read_file(series_dir + file_name, stop_before_pixels=True)

    meta_dict = {}

    meta_dict['PatientID'] = dicom_data.PatientID
    meta_dict['StudyID'] = dicom_data.StudyID
    meta_dict['StudyInstanceUID'] = dicom_data.StudyInstanceUID
    meta_dict['Modality'] = dicom_data.Modality

    if dicom_data.Modality in ['PT', 'CT']:
        # Using sitk to extract image mata-data
        meta_dict['Spacing'] = sitk_image.GetSpacing()
        meta_dict['Width'] = sitk_image.GetWidth()
        meta_dict['Height'] = sitk_image.GetHeight()
        meta_dict['Depth'] = sitk_image.GetDepth()
        meta_dict['Direction'] = sitk_image.GetDirection()

    if dicom_data.Modality == 'RTSTRUCT':
        meta_dict['StructureSetLabel'] = dicom_data.StructureSetLabel
        meta_dict['StructureSetName'] = dicom_data.StructureSetName

    return sitk_image, meta_dict



def read_image(file_path, print_meta=True, print_stats=False):
    """
    Read NIfTI/NRRD file to sitk image
    """
    sitk_image = sitk.ReadImage(file_path)

    if print_meta:
        print("Loaded image:", file_path.split('/')[-1])
        print("Patient ID:", file_path.split('/')[-1].split('_')[0])

        if '_gtvt' in file_path:
            modality = 'Binary GTV mask'
            sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
        elif '_ct' in file_path: modality = 'CT'
        elif '_pt' in file_path: modality = 'PT'
        print("Modality:", modality)

        image_size = sitk_image.GetSize()
        pixel_spacing = sitk_image.GetSpacing()
        print("Image size:", image_size)
        print("Pixel spacing (mm):", pixel_spacing)
        print("Physical size (mm):", [image_size[i]*pixel_spacing[i] for i in range(3)])
        #print("Components per pixel:", sitk_image.GetNumberOfComponentsPerPixel())

    if print_stats:
        image_stats = sitk.StatisticsImageFilter()
        image_stats.Execute(sitk_image)

        print(f"\n----- Image Statistics ----- \n Max Intensity: {image_stats.GetMaximum()} \
                \n Min Intensity: {image_stats.GetMinimum()} \n Mean: {image_stats.GetMean()} \
                \n Variance: {image_stats.GetVariance()} \n")

    print("\n")
    return sitk_image