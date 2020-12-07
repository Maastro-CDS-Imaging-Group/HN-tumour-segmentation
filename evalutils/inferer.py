import os
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import SimpleITK as sitk

from datautils.conversion import *
from datautils.patch_aggregation import PatchAggregator3D, get_pred_patches_list
from evalutils.metrics import dice

logging.basicConfig(level=logging.DEBUG)


class Inferer():

    def __init__(self, 
                 model,                
                 volume_loader, patch_sampler, patch_aggregator,
                 hardware_config, input_data_config, inference_config):
        
        self.hardware_config = hardware_config
        self.device = self.hardware_config['device']

        self.model = model.to(self.device)
        if self.hardware_config['enable-distributed']:
            self.model = torch.nn.DataParallel(self.model)
            checkpoint_dict = torch.load(f"{inference_config['model-filepath']}")
            model_state_dict = checkpoint_dict['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.nn_name = self.model.module.nn_name
        else:
            checkpoint_dict = torch.load(f"{inference_config['model-filepath']}", map_location=self.device)
            model_state_dict = checkpoint_dict['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.nn_name = self.model.nn_name
        self.model.eval()

        self.softmax = torch.nn.Softmax(dim=1) # Softmax along channel dimension
        logging.debug(f"Loaded model: {inference_config['model-filepath']}")

        self.volume_loader = volume_loader
        self.patch_sampler = patch_sampler
        self.patch_aggregator = patch_aggregator
 
        # Configs
        self.input_data_config = input_data_config
        self.inference_config = inference_config

        with open(self.inference_config['patient-id-filepath'], 'r') as pf:
            self.patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']
        center = self.inference_config['subset-name'].split('-')[1] 
        self.patient_ids = [p_id for p_id in self.patient_ids if center in p_id]

        # Output saving stuff
        os.makedirs(f"{self.inference_config['output-save-dir']}/predicted", exist_ok=True)
        if self.nn_name == 'msam3d':
            os.makedirs(f"{self.inference_config['output-save-dir']}/attention_maps", exist_ok=True)


    def run_inference(self):
        dice_scores = {}
        avg_dice = 0
        for i, patient_dict in enumerate(tqdm(self.volume_loader)):
            # Run one inference step - split patient volumes into patches, run forward pass with the patches, 
            # aggregate prediction patches into full volume labelmap and compute dice score
            patient_pred_volume, patient_dice_score, attention_map_volume = self._inference_step(patient_dict)
        
            # Save the results -- pred volumes in NRRD, dice in CSV, attention maps in NRRD (if enabled)
            p_id = self.patient_ids[i]
            if self.inference_config['save-results']:
                pred_volume_sitk = np2sitk(patient_pred_volume, has_whd_ordering=False)
                output_nrrd_filename = f"{p_id}_pred_gtvt.nrrd"
                sitk.WriteImage(pred_volume_sitk, f"{self.inference_config['output-save-dir']}/predicted/{output_nrrd_filename}", True)

                # To save attention map, in case of MSAM3D
                if self.nn_name == 'msam3d' and attention_map_volume is not None:                
                    attention_map_volume_sitk = np2sitk(attention_map_volume, has_whd_ordering=False)
                    output_nrrd_filename = f"{p_id}_am.nrrd"
                    sitk.WriteImage(attention_map_volume_sitk, f"{self.inference_config['output-save-dir']}/attention_maps/{output_nrrd_filename}", True)
           
            if self.inference_config['compute-metrics']:
                dice_scores[p_id] = patient_dice_score
                avg_dice += patient_dice_score
        
        if self.inference_config['compute-metrics']:
            avg_dice /= len(self.volume_loader)
            dice_scores['average'] = avg_dice
            df = pd.DataFrame.from_dict(dice_scores, orient="index")
            logging.debug(df)
            if self.inference_config['save-results']:
                df.to_csv(f"{self.inference_config['output-save-dir']}/dice_scores.csv")

        logging.debug(f"Saved results to: {self.inference_config['output-save-dir']}")


    def _inference_step(self, patient_dict):
        # Remove the batch dimension from input and target volumes of the patient dict
        for key, value in patient_dict.items():
            patient_dict[key] = value[0]

        # Get full list of patches
        patches_list = self.patch_sampler.get_samples(patient_dict, num_patches=self.inference_config['valid-patches-per-volume'])

        # Stuff to accumulate
        patient_pred_patches_list = []
        attention_map_patches_list = []  # Only used when enabled 

        with torch.no_grad(): # Disable autograd

            # Take batch_of_patches_size number of patches at a time and push through the network
            valid_patches_per_volume = self.inference_config['valid-patches-per-volume']
            batch_of_patches_size = self.inference_config['batch-of-patches-size']
            batch_of_patches_size = min(valid_patches_per_volume, batch_of_patches_size)

            for p in range(0, valid_patches_per_volume, batch_of_patches_size):

                # Get input patches
                if self.input_data_config['is-bimodal']: # In case of bimodal input                    
                    if self.input_data_config['input-representation'] == 'separate-volumes': # For PET and CT as separate volumes
                        PET_patches = torch.stack([patches_list[p+i]['PET'] for i in range(batch_of_patches_size)], dim=0).to(self.device)
                        CT_patches = torch.stack([patches_list[p+i]['CT'] for i in range(batch_of_patches_size)], dim=0).to(self.device)                        
                        input_patches = [PET_patches, CT_patches] # Pack these tensors into a list
                    if self.input_data_config['input-representation'] == 'multichannel-volume': # For PET and CT as a single 2-channel volume
                        input_patches = torch.stack([patches_list[p+i]['PET-CT'] for i in range(batch_of_patches_size)], dim=0).to(self.device)                
                else: # In case of unimodal input 
                    modality = self.input_data_config['input-modality']
                    input_patches = torch.stack([patches_list[p+i][modality] for i in range(batch_of_patches_size)], dim=0).to(self.device)
                
                # Get ground truth labelmap
                if self.inference_config['compute-metrics']:
                    target_labelmap_patches = torch.stack([patches_list[p+i]['target-labelmap'] for i in range(batch_of_patches_size)], dim=0).long().to(self.device)

                # Forward pass
                if self.nn_name == 'msam3d':
                    pred_patches, attention_map_patches = self.model(input_patches)
                else: 
                    pred_patches = self.model(input_patches)
                pred_patches = self.softmax(pred_patches)
                
                # Convert the predicted batch of probabilities to a list of labelmap patches (or foreground probabilities), and store
                patient_pred_patches_list.extend(get_pred_patches_list(pred_patches, as_probabilities=self.inference_config['save-as-probabilities'])) 
                
                if self.nn_name == 'msam3d' and attention_map_patches is not None:
                    attention_map_patches_list.extend(get_pred_patches_list(attention_map_patches, as_probabilities=True)) 


            # Aggregate into full volume
            patient_pred_volume = self.patch_aggregator.aggregate(patient_pred_patches_list, device=self.device) 
            

            if self.nn_name == 'msam3d' and attention_map_patches is not None:
                attention_map_volume = self.patch_aggregator.aggregate(attention_map_patches_list, device=self.device) 
                attention_map_volume = attention_map_volume.cpu().numpy()
            else:
                attention_map_volume = None

        # Compute metrics, if needed
        if self.inference_config['compute-metrics']:
            if self.inference_config['save-as-probabilities']:
                patient_pred_labelmap = (patient_pred_volume >= 0.5).long().cpu().numpy()
                patient_dice_score = dice(patient_pred_labelmap, patient_dict['target-labelmap'].cpu().numpy())
            else:
                patient_dice_score = dice(patient_pred_volume.cpu().numpy(), patient_dict['target-labelmap'].cpu().numpy())
        else:
            patient_dice_score = None

        patient_pred_volume = patient_pred_volume.cpu().numpy()
        return patient_pred_volume, patient_dice_score, attention_map_volume