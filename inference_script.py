"""
Script to generate GTV segmentation outputs given a trained model and a validation set.

    - The outputs could be either GTV foreground probability maps or binary labelmaps, depending on user's shoice
    - Optionally, avg Dice metric can be computed
    - Optionally, the model outputs and the metrics can be saved to disk

"""

import argparse
import logging
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
from evalutils.inferer import Inferer
import config_utils



# Constants
DEFAULT_DATA_CONFIG_FILE = "./config_files/data-crS_rs113-petct_default.yaml"
DEFAULT_NN_CONFIG_FILE = "./config_files/nn-unet3d_default.yaml"
DEFAULT_INFERENCE_CONFIG_FILE = "./config_files/infer-default.yaml"



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
	parser.add_argument("--infer_config_file",
	                    type=str,
						help="Path to the trainval config file",
						default=DEFAULT_INFERENCE_CONFIG_FILE)
				

	args = parser.parse_args()
	return args


def main(global_config):
    # -----------------------------------------------
    # Data pipeline
    # -----------------------------------------------

    # Dataset
    preprocessor = Preprocessor(**global_config['preprocessor-kwargs'])

    if not global_config['inferer-kwargs']['input_data_config']['is-bimodal']:
        dataset = HECKTORUnimodalDataset(**global_config['dataset-kwargs'], preprocessor=preprocessor)
    else:
        dataset = HECKTORPETCTDataset(**global_config['dataset-kwargs'], preprocessor=preprocessor)


    # Patch based inference stuff
    volume_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    patch_sampler = PatchSampler3D(**global_config['patch-sampler-kwargs'])
    patch_aggregator = PatchAggregator3D(**global_config['patch-aggregator-kwargs'])
      

    # -----------------------------------------------
    # Network
    # -----------------------------------------------

    if global_config['nn-name'] == "unet3d":
        model = nnmodules.UNet3D(**global_config['nn-kwargs'])

    elif global_config['nn-name'] == "msam3d":
        model = nnmodules.MSAM3D(**global_config['nn-kwargs'])


    # -----------------------------------------------
    # Inference
    # -----------------------------------------------

    inferer = Inferer(model,
                    volume_loader, patch_sampler, patch_aggregator,
                    **global_config['inferer-kwargs'])

    inferer.run_inference()


if __name__ == '__main__':
    cli_args = get_cli_args()
    global_config = config_utils.build_config(cli_args, training=False)

    main(global_config)