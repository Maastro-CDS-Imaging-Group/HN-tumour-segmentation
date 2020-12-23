"""
Conducts a comprehensive performance evaluation of a given segmentation approach under the cross-validation setting.
Takes predicted and ground truth labelmaps as input, and generates a performance scorecard. 

Results include: 
    - Per centre metrics: Centre avg Dice, Centre avg IoU
    - Global: Crossval avg Dice, Crossval avg IoU, SPP, model complexity

Additional output files:
	- SPP plot
	- Per patient metrics CSV - Includes Hausdorff
	
"""

import os
import argparse
import logging
from collections import OrderedDict
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt

from datautils.conversion import *
import nnmodules
import evalutils.metrics as metrics
from evalutils.stats import SPP
from evalutils.perf_scorecard import PerformanceScorecard



DATA_ROOT_DIR = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train"
SAVED_MODEL_ROOT_DIR = "/home/zk315372/Chinmay/saved_models"
MODEL_PREDS_ROOT_DIR = "/home/zk315372/Chinmay/model_predictions"
OUTPUT_ROOT_DIR = "/home/zk315372/Chinmay/model_performances"

PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

DEFAULT_DATASET_NAME = "hecktor-crS_rs113"
DEFAULT_NN_NAME = "unet3d"
DEFAULT_MODEL_INPUT_INFO = "latefusion"
DEFAULT_NN_CONFIG_FILE = "./config_files/nn-unet3d_unimodal.yaml"


def get_cli_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--models_root_dir",
	                    type=str,
						help="Path to the saved models root directory",
						default=SAVED_MODEL_ROOT_DIR)
	parser.add_argument("--model_preds_root_dir",
	                    type=str,
						help="Path to the model predictions root directory",
						default=MODEL_PREDS_ROOT_DIR)
	parser.add_argument("--data_root_dir",
	                    type=str,
						help="Path to the dataset root directory",
						default=DATA_ROOT_DIR)
	parser.add_argument("--patient_id_filepath",
	                    type=str,
						help="Path to the patient IDs file",
						default=PATIENT_ID_FILEPATH)
	parser.add_argument("--output_root_dir",
	                    type=str,
						help="Path to the output root directory",
						default=OUTPUT_ROOT_DIR)

	parser.add_argument("--dataset_name",
	                    type=str,
						default=DEFAULT_DATASET_NAME)
	parser.add_argument("--nn_name",
	                    type=str,
						default=DEFAULT_NN_NAME)
	parser.add_argument("--model_input_info",
						type=str,
						default=DEFAULT_MODEL_INPUT_INFO)
	parser.add_argument("--nn_config_file",
	                    type=str,
						help="Path to the network config file",
						default=DEFAULT_NN_CONFIG_FILE)
	
	args = parser.parse_args()
	return args


def main(args):
	data_dir = f"{args.data_root_dir}/{args.dataset_name.split('-')[1]}_hecktor_nii"
	output_dir = f"{args.output_root_dir}/{args.dataset_name}/{args.nn_name}_{args.model_input_info}"
	os.makedirs(output_dir, exist_ok=True)
	# output_dir = "./temp_dir" ##

	with open(args.patient_id_filepath, 'r') as pf:
		patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

	# Correction for crS
	print("Warning -- Correction for crop-S data: Not considering patients CHUM010 and CHUS021 in evaluation\n")
	if "crS_rs113" in data_dir:
		patient_ids.remove("CHUM010")
		patient_ids.remove("CHUS021")


	# Define the scorecard instance
	scorecard = PerformanceScorecard(output_dir=output_dir)
	
	scorecard.add_info(info_name="Dataset Code", info=args.dataset_name, category="Data")


	# -------------------------------------------------------------------
	# Get model details
	print("Getting segmentation model details ...")
	
	scorecard.add_info(info_name="Network Architecture", info=args.nn_name, category="Model")
	scorecard.add_info(info_name="Model Input", info=args.model_input_info, category="Model")
	
	with open(args.nn_config_file, 'r') as nnc:
		yaml_nn_config = yaml.safe_load(nnc)

	if args.nn_name == "unet3d":
		model = nnmodules.UNet3D(**yaml_nn_config['nn-kwargs'])
	elif args.nn_name == "msam3d":
		model = nnmodules.MSAM3D(**yaml_nn_config['nn-kwargs'])
	
	n_trainable_params = count_model_parameters(model)
	scorecard.add_info(info_name="Trainable Parameters", info=n_trainable_params, category="Model")
	
	print("Number of trainable parameters:", n_trainable_params)
	print()


	# -------------------------------------------------------------------
	# Compute centre-wise similarity metrics  
	print("Computing similarity metrics ...")
	centre_ids = ('CHGJ', 'CHMR', 'CHUM', 'CHUS')
	
	centre_dice_dict = {}
	centre_jaccard_dict = {}
	patient_volume_overlap_dict = OrderedDict()  # To contain volume overlap metrics (per patient)
	patient_surface_distance_dict = OrderedDict() # To contain surface distance metrics (per patient)
	
	for centre in centre_ids: # For each centre
		
		centre_patient_ids = [p_id for p_id in patient_ids if centre in p_id]
		preds_dir = f"{args.model_preds_root_dir}/{args.dataset_name}/{args.nn_name}_{args.model_input_info}/crossval-{centre}/predicted"
		
		centre_dice = 0
		centre_jaccard = 0

		for p_id in centre_patient_ids: # For each patient in this centre
			gtv_labelmap = sitk2np(sitk.ReadImage(f"{data_dir}/{p_id}_ct_gtvt.nii.gz"), keep_whd_ordering=True)
			pred_labelmap = sitk2np(sitk.ReadImage(f"{preds_dir}/{p_id}_pred_gtvt.nrrd"), keep_whd_ordering=True)

			# If predictions are as foreground probabilities, convert to a binary labelmap
			if list(np.unique(pred_labelmap)) != [0, 1]:
				pred_labelmap = (pred_labelmap >= 0.5).astype(np.int8)

			# Compute metrics
			dice_score = metrics.dice(pred_labelmap, gtv_labelmap)
			jaccard_score, intersection, union = metrics.jaccard(pred_labelmap, gtv_labelmap, return_i_and_u=True)
			hausdorff_distance = metrics.hausdorff(pred_labelmap, gtv_labelmap, dim_ordering='whd')
			# hausdorff_distance = 0 ##

			# Accumulate
			patient_volume_overlap_dict[p_id] = [dice_score, jaccard_score, intersection, union]
			centre_dice += dice_score
			centre_jaccard += jaccard_score
			patient_surface_distance_dict[p_id] = [hausdorff_distance]
		
		centre_dice /= len(centre_patient_ids)
		centre_dice_dict[centre] = round(float(centre_dice), 5)

		centre_jaccard /= len(centre_patient_ids)
		centre_jaccard_dict[centre] = round(float(centre_jaccard), 5)

	scorecard.add_info(info_name="Centre Average Dice", info=centre_dice_dict, category="Per Centre Metrics")
	scorecard.add_info(info_name="Centre Average Jaccard", info=centre_jaccard_dict, category="Per Centre Metrics")

	print("Per centre Dice:", centre_dice_dict)
	print("Per centre Jaccard:", centre_jaccard_dict)
	print()
	

	# -------------------------------------------------------------------
	# Global metrics (i.e. over all patients)
	global_avg_dice = np.mean(np.array([value[0] for value in patient_volume_overlap_dict.values()]))
	global_avg_jaccard = np.mean(np.array([value[1] for value in patient_volume_overlap_dict.values()]))

	scorecard.add_info(info_name="Global Average Dice", info=round(float(global_avg_dice), 5), category="Global Metrics")
	scorecard.add_info(info_name="Global Average Jaccard", info=round(float(global_avg_jaccard), 5), category="Global Metrics")

	print("Global average Dice:", round(global_avg_dice, 5))
	print("Global average Jaccard:", round(global_avg_jaccard, 5))
	print()


	# -------------------------------------------------------------------
	# Compute SPP (global, over all patients)
	print("Computing SPP ...")
	spp = SPP(patient_ids, output_dir)
	perf_distribution_info = spp.estimate_performance(patient_volume_overlap_dict)	
	
	scorecard.add_info(info_name="SPP", info=perf_distribution_info, category="Global Metrics")

	print(f"alpha: {perf_distribution_info['alpha']:.5f}, beta: {perf_distribution_info['beta']:.5f}")
	print(f"Performance mean: {perf_distribution_info['performance-mean']:.5f}, Performance stddev: {perf_distribution_info['performance-stddev']:.5f}\n")
	print()


	# -------------------------------------------------------------------
	# Write results into output dir
	print("Saving results in output directory ...")
	scorecard.write_to_file()
	
	spp.plot_performance()

	per_patient_metrics = OrderedDict()
	for p_id in patient_ids:
		per_patient_metrics[p_id] = patient_volume_overlap_dict[p_id]
		per_patient_metrics[p_id].extend(patient_surface_distance_dict[p_id])
		# print(per_patient_metrics[p_id])

	df = pd.DataFrame.from_dict(per_patient_metrics, orient='index', columns=['Dice', 'Jaccard', 'Intersection', 'Union', 'Hausdorff'])
	df.to_csv(f"{output_dir}/per_patient_metrics.csv")
	print("Saved results to:", output_dir)


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = get_cli_args()
    main(args)