import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

from datautils.conversion import *
import evalutils.metrics as metrics


DATA_DIR = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crFHN_rs113_hecktor_nii"
CENTRE_ID = "CHUM"
PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

PREDS_DIR = "/home/zk315372/Chinmay/model_predictions/hecktor-crFHN_rs113/msam3d_petct/patch_sample-pet_weighted/predicted"
OUTPUT_DIR = "/home/zk315372/Chinmay/model_performances/hecktor-crFHN_rs113/msam3d_petct/patch_sample-pet_weighted"


def main():
    data_dir = DATA_DIR
    preds_dir = PREDS_DIR
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    # output_dir = "./temp_dir" ##

    with open(PATIENT_ID_FILEPATH, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']
    centre_patient_ids = [p_id for p_id in patient_ids if CENTRE_ID in p_id]
    
    patient_dice_dict = {}
    avg_dice = 0

    for p_id in tqdm(centre_patient_ids): # For each patient in this centre
        # Fetch the labelmaps
        gtv_labelmap = sitk2np(sitk.ReadImage(f"{data_dir}/{p_id}_ct_gtvt.nii.gz"), keep_whd_ordering=True).astype(np.int8)
        pred_labelmap = sitk2np(sitk.ReadImage(f"{preds_dir}/{p_id}_pred_gtvt.nrrd"), keep_whd_ordering=True).astype(np.int8)

        # Compute metrics
        dice_score = metrics.dice(pred_labelmap, gtv_labelmap)

        # Accumulate
        patient_dice_dict[p_id] = dice_score
        avg_dice += dice_score

    avg_dice /= len(centre_patient_ids)
    patient_dice_dict['average'] = avg_dice
    df = pd.DataFrame.from_dict(patient_dice_dict, orient="index")
    print(df)
    df.to_csv(f"{output_dir}/per_patient_metrics.csv")



if __name__ == '__main__':
    main()