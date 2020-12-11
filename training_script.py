import logging, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datautils.preprocessing import Preprocessor
from datasets.hecktor_unimodal_dataset import HECKTORUnimodalDataset
from datasets.hecktor_petct_dataset import HECKTORPETCTDataset
from datautils.patch_sampling import PatchSampler3D, PatchQueue
from datautils.patch_aggregation import PatchAggregator3D
import nnmodules
from trainutils.trainer import Trainer
import config_utils

# Reproducibility settings
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False

# Constants
DEFAULT_DATA_CONFIG_FILE = "./config_files/data-crFHN_rs113-petct_default.yaml"
DEFAULT_NN_CONFIG_FILE = "./config_files/nn-msam3d_default.yaml"
DEFAULT_TRAINVAL_CONFIG_FILE = "./config_files/trainval-default.yaml"


def get_cli_args():
	parser = argparse.ArgumentParser()
	
	# Config filepaths
	parser.add_argument("--data_config_file",
	                    type=str,
						help="Path to the data config file",
						default=DEFAULT_DATA_CONFIG_FILE)
	parser.add_argument("--nn_config_file",
	                    type=str,
						help="Path to the network config file",
						default=DEFAULT_NN_CONFIG_FILE)
	parser.add_argument("--trainval_config_file",
	                    type=str,
						help="Path to the trainval config file",
						default=DEFAULT_TRAINVAL_CONFIG_FILE)
				

	# Overrides
	parser.add_argument("--run_name",
	                    type=str,
						help="Name of the run",
						default="trial-run")


	args = parser.parse_args()
	return args


def main(global_config):
	# -----------------------------------------------
	# Data pipeline
	# -----------------------------------------------

	# Datasets
	preprocessor = Preprocessor(**global_config['preprocessor-kwargs'])

	if not global_config['trainer-kwargs']['input_data_config']['is-bimodal']:
		train_dataset = HECKTORUnimodalDataset(**global_config['train-dataset-kwargs'], preprocessor=preprocessor)
		val_dataset = HECKTORUnimodalDataset(**global_config['val-dataset-kwargs'], preprocessor=preprocessor)
	else:
		train_dataset = HECKTORPETCTDataset(**global_config['train-dataset-kwargs'], preprocessor=preprocessor)
		val_dataset = HECKTORPETCTDataset(**global_config['val-dataset-kwargs'], preprocessor=preprocessor)

	# Patch based training stuff
	train_sampler = PatchSampler3D(**global_config['train-patch-sampler-kwargs'])
	train_patch_queue = PatchQueue(**global_config['train-patch-queue-kwargs'], dataset=train_dataset, sampler=train_sampler)
	train_patch_loader = DataLoader(train_patch_queue, **global_config['train-patch-loader-kwargs'])
	
	# Patch based inference stuff
	val_sampler = PatchSampler3D(**global_config['val-patch-sampler-kwargs'])
	val_aggregator = PatchAggregator3D(**global_config['val-patch-aggregator-kwargs'])
	val_volume_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


	# -----------------------------------------------
	# Network
	# -----------------------------------------------

	if global_config['nn-name'] == "unet3d":
		model = nnmodules.UNet3D(**global_config['nn-kwargs'])

	elif global_config['nn-name'] == "msam3d":
		model = nnmodules.MSAM3D(**global_config['nn-kwargs'])
		

	# -----------------------------------------------
	# Training
	# -----------------------------------------------

	trainer = Trainer(model,
					 train_patch_loader, val_volume_loader, val_sampler, val_aggregator,
					 **global_config['trainer-kwargs'])

	trainer.run_training()


if __name__ == '__main__':
	cli_args = get_cli_args()
	global_config = config_utils.build_config(cli_args, training=True)

	main(global_config)