import os
import numpy as np
import SimpleITK as sitk

from datautils.conversion import *


PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"
MODEL_PREDS_ROOT_DIR = "/home/zk315372/Chinmay/model_predictions"


def main():
    with open(PATIENT_ID_FILEPATH, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

    # Correction for crS
    print("Warning -- Correction for crop-S data: Not considering patients CHUM010 and CHUS021 in evaluation\n")
    patient_ids.remove("CHUM010")
    patient_ids.remove("CHUS021")

    centre_ids = ('CHGJ', 'CHMR', 'CHUM', 'CHUS')

    for centre in centre_ids:
        centre_patient_ids = [p_id for p_id in patient_ids if centre in p_id]
        
        unet3d_pet_preds_dir = f"{MODEL_PREDS_ROOT_DIR}/hecktor-crS_rs113/unet3d_pet/crossval-{centre}/predicted"
        unet3d_ct_preds_dir = f"{MODEL_PREDS_ROOT_DIR}/hecktor-crS_rs113/unet3d_ct/crossval-{centre}/predicted"
        
        latefusion_preds_dir = f"{MODEL_PREDS_ROOT_DIR}/hecktor-crS_rs113/unet3d_latefusion/crossval-{centre}/predicted"
        os.makedirs(latefusion_preds_dir, exist_ok=True)
        
        for p_id in centre_patient_ids:
            unet3d_pet_pred = sitk2np(sitk.ReadImage(f"{unet3d_pet_preds_dir}/{p_id}_pred_gtvt.nrrd"), keep_whd_ordering=True)
            unet3d_ct_pred = sitk2np(sitk.ReadImage(f"{unet3d_ct_preds_dir}/{p_id}_pred_gtvt.nrrd"), keep_whd_ordering=True)

            fused_pred_labelmap = fuse_predictions(unet3d_pet_pred, unet3d_ct_pred)

            fused_pred_sitk = np2sitk(fused_pred_labelmap, has_whd_ordering=True)
            sitk.WriteImage(fused_pred_sitk, f"{latefusion_preds_dir}/{p_id}_pred_gtvt.nrrd", True)


def fuse_predictions(pet_pred, ct_pred):
    '''
    Voxel wise average followed by thresholding
    '''
    fused_pred = (pet_pred + ct_pred) / 2
    fused_pred_labelmap = (fused_pred >= 0.5).astype(np.int8) 
    return fused_pred_labelmap


if __name__ == '__main__':
    main()