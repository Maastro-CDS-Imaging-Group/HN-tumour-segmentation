import numpy as np
import SimpleITK as sitk

data_dir = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crFHN_rs113_hecktor_nii"
patient_id_filepath = "./hecktor_meta/patient_IDs_train.txt"


with open(patient_id_filepath, 'r') as pf:
    patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

for p_id in patient_ids:
    gtv_file = f"{data_dir}/{p_id}_ct_gtvt.nii.gz"
    gtv_sitk = sitk.ReadImage(gtv_file)
    gtv_np = sitk.GetArrayFromImage(gtv_sitk)
    print(p_id, np.unique(gtv_np))