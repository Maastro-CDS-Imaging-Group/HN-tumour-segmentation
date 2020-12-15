"""
Performs a comprehensive evaluation of a given segmentation approach.
Takes predicted and ground truth labelmaps as input, and generates a performance scorecard. 
Results include: 
    - Per centre metrics: Centre avg Dice, Centre avg IoU
    - Global: Crossval avg Dice, Dice vs. patient plot, Crossval avg IoU, SPP, computation time, model complexity
"""

import argparse
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt

from datautils.conversion import *
from evalutils.metrics import dice, jaccard
from evalutils.stats import SPP



DATA_ROOT_DIR = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train"
SAVED_MODEL_ROOT_DIR = "/home/zk315372/Chinmay/saved_models"
MODEL_PREDS_ROOT_DIR = "/home/zk315372/Chinmay/model_predictions"
OUTPUT_ROOT_DIR = "/home/zk315372/Chinmay/performance_scorecards"

PATIENT_ID_FILEPATH = "./hecktor_meta/patient_IDs_train.txt"

DEFAULT_DATASET_NAME = "hecktor-crS_rs113"
DEFAULT_NN_NAME = "unet3d"
DEFAULT_MODEL_INPUT_INFO = "petct"


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
	
	args = parser.parse_args()
	return args


def main(args):
	per_centre_metrics = {}
	global_metrics = {}

	data_dir = f"{args.data_root_dir}/{args.dataset_name.split('-')[1]}_hecktor_nii"
	# output_dir = f"{args.output_root_dir}/{args.dataset_name}/{args.nn_name}_{args.model_input_info}"
	output_dir = "./temp_dir"

	with open(args.patient_id_filepath, 'r') as pf:
		patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

	# Correction for crS
	print("Correction for crop-S data: Not considering patients CHUM010 and CHUS021 in evaluation\n")
	if "crS_rs113" in data_dir:
		patient_ids.remove("CHUM010")
		patient_ids.remove("CHUS021")

	# -------------------------------------------------------------------
	# Compute centre-wise similarity metrics  
	print("Computing volume overlap metrics ...")
	centre_ids = ('CHGJ', 'CHMR', 'CHUM', 'CHUS')
	
	patient_dice_dict = OrderedDict()
	patient_jaccard_dict = OrderedDict()
	
	for centre in centre_ids: # For each centre
		
		centre_patient_ids = [p_id for p_id in patient_ids if centre in p_id]
		preds_dir = f"{args.model_preds_root_dir}/{args.dataset_name}/{args.nn_name}_{args.model_input_info}/crossval-{centre}/predicted"
		
		centre_dice = 0

		for p_id in centre_patient_ids: # For each patient in this centre
			gtv_labelmap = sitk2np(sitk.ReadImage(f"{data_dir}/{p_id}_ct_gtvt.nii.gz"), keep_whd_ordering=True)
			pred_labelmap = sitk2np(sitk.ReadImage(f"{preds_dir}/{p_id}_pred_gtvt.nrrd"), keep_whd_ordering=True)

			# if np.unique(pred_labelmap) != [0,1]:
			# 	pred_labelmap = pred_labelmap[pred_labelmap >= 0.5].astype(np.int8)

			# Compute metrics
			dice_score = dice(pred_labelmap, gtv_labelmap)
			iou, intersection, union = jaccard(pred_labelmap, gtv_labelmap, return_i_and_u=True)

			# Accumulate
			centre_dice += dice_score
			patient_dice_dict[p_id] = dice_score
			patient_jaccard_dict[p_id] = (iou, intersection, union)
			
		centre_dice /= len(centre_patient_ids)
		per_centre_metrics[centre] = {'dice': centre_dice}

	print(f"Per centre metrics: {per_centre_metrics}\n")
	
	# -------------------------------------------------------------------
	# Compute SPP (global)
	print("Computing SPP ...")
	spp = SPP(patient_ids, output_dir)
	alpha, beta, perf_mean, perf_stddev = spp.estimate_performance(patient_jaccard_dict)	
	print(f"alpha: {alpha}, beta: {beta}")
	print(f"Performance mean: {perf_mean}, Performance stddev: {perf_stddev}\n")

	# -------------------------------------------------------------------
	# Get model details
	print("Running model speed test ...")
	# TODO
	n_trainable_params = None
	infer_time = None
	gpu_details = None
	print(f"Number of trainable parameters: {n_trainable_params}")
	print(f"Average inference time: {infer_time}")
	print(f"GPU hardware: {gpu_details}")

	# -------------------------------------------------------------------
	# Write results into output dir
	print("Saving results in output directory ...")
	# TODO
	scorecard_string = construct_scorecard()
	with open(f"{data_dir}/perf_scorecard.txt", 'w') as sf:
		sf.write(scorecard_string)

	spp.plot_performance()
	df = pd.DataFrame.from_dict(patient_dice_dict, orient='index', columns=['Dice'])
	df.to_csv(f"{output_dir}/dice_scores.csv")
	print(f"Saved results to: {output_dir}")


def construct_scorecard():
	# TODO
	scorecard_string = None
	return scorecard_string


if __name__ == '__main__':
    args = get_cli_args()
    main(args)