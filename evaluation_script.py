"""
Performs a comprehensive evaluation of a given segmentation approach.
Takes predicted and ground truth labelmaps as input, and generates a performance scorecard. 
Results include: 
    - Per centre metrics: Centre avg Dice, Centre avg IoU, Centre avg Hausdorff
    - Global: Crossval avg Dice, Crossval avg IoU, Crossval avg Hausdorff, SPP, computation time, model complexity
"""

import argparse
import logging
import numpy as np
from tqdm import tqdm


DEFAULT_MODEL_FILEPATH = None
DEFAULT_OUTPUTS_DIR = None
DATA_DIR = None
PATIENT_ID_FILEPATH = None


def get_cli_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--model_filepath",
	                    type=str,
						help="Path to the saved .pt model",
						default=DEFAULT_MODEL_FILEPATH)
	parser.add_argument("--model_outputs_dir",
	                    type=str,
						help="Path to the model predictions",
						default=DEFAULT_OUTPUTS_DIR)
	parser.add_argument("--data_dir",
	                    type=str,
						help="Path to the dataset",
						default=DATA_DIR)
	parser.add_argument("--patient_id_filepath",
	                    type=str,
						help="Path to the patient IDs file",
						default=PATIENT_ID_FILEPATH)
				
	args = parser.parse_args()
	return args


def main(args):
    pass



if __name__ == '__main__':
    args = get_cli_args()
    main(args)