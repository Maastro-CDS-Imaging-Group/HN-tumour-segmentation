import os
import logging
import numpy as np
import torch
from tqdm import tqdm
import wandb

from trainutils.loss_functions import build_loss_function
from datautils.patch_aggregation import PatchAggregator3D, get_pred_labelmap_patches_list
from inferutils.metrics import volumetric_dice

logging.basicConfig(level=logging.DEBUG)



class Trainer():

    def __init__(self, 
                model, 
                train_patch_loader, val_volume_loader, val_sampler, val_aggregator,
                device,
                input_data_config, training_config, validation_config, logging_config):
        
        self.model = model
        self.train_patch_loader = train_patch_loader
        self.val_volume_loader = val_volume_loader
        self.val_sampler = val_sampler
        self.val_aggregator = val_aggregator

        self.device = device 

        # Config
        self.input_data_config = input_data_config
        self.training_config = training_config
        self.validation_config = validation_config
        self.logging_config = logging_config

        self.criterion = build_loss_function(training_config['loss-name'], self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.training_config['learning-rate'])
        
        self.start_epoch = 1

        self.checkpoint_dir = f"{self.training_config['checkpoint-root-dir']}/{self.logging_config['run-name']}"
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        if training_config['continue-from-checkpoint']:
            # Checkpoint name example - unet3d_pet_e005.pt
            self.model.load_state_dict(torch.load(f"{self.checkpoint_dir}/{training_config['checkpoint-filename']}"))
            self.start_epoch = int(training_config['checkpoint-filename'].split('.')[0][-3:]) + 1


        # Logging related
        if self.logging_config['enable-wandb']:
            wandb.init(entity=self.logging_config['wandb-entity'],
			           project=self.logging_config['wandb-project'],
			           name=self.logging_config['run-name'],
			           config=self.logging_config['wandb-config']
			          )
            wandb.config.update({'dataset-name': self.training_config['dataset-name'],
                                 'train-subset-name': self.training_config['subset-name'],
                                 'val-subset-name': self.validation_config['subset-name'],
                                'batch-of-patches-size': self.validation_config['batch-of-patches-size'],
                                'learning-rate': self.training_config['learning-rate'],
                                'start-epoch': self.start_epoch,
                                'num-epochs': self.start_epoch + self.training_config['num-epochs']})
            wandb.watch(self.model)

        # Log some stuff
        logging.debug("Trainer initialized")
        logging.debug(f"Train subset name: {self.training_config['train-subset-name']}")
        logging.debug(f"Validation subset name: {self.validation_config['val-subset-name']}")


    def run_training(self):
        """
        Training loop
        """
        
        logging.debug(f"RUN NAME: {self.logging_config['run-name']}")

        if self.training_config['continue-from-checkpoint']:
            logging.debug(f"Loading checkpoint: {self.training_config['checkpoint-filename']}")
            logging.debug(f"Continuing from epoch {self.start_epoch}")

        
        for epoch in range(self.start_epoch, self.start_epoch + self.training_config['num-epochs']):
            logging.debug(f"Epoch {epoch}")

            epoch_train_loss = 0
            epoch_val_loss = 0
            epoch_val_dice = 0

            # Train --
            self.model.train() # Set the model in train mode
            logging.debug("Training ...")
            for batch_of_patches in tqdm(self.train_patch_loader):

                # Run one train step -- forward + backward once
                train_loss = self._train_step(batch_of_patches)

                # Accumulate loss value
                epoch_train_loss += train_loss
                
            epoch_train_loss /= len(self.train_patch_loader)

            # Clear CUDA cache
            torch.cuda.empty_cache()


            # Validate --
            logging.debug("Validating ...")
            self.model.eval() # Set the model in inference mode

            # Iterate over patients in validation set
            for patient_dict in tqdm(self.val_volume_loader):
                # Run one validation step - split patient volumes into patches, run forward pass with the patches, 
                # aggregate prediction patches into full volume labelmap and compute dice score
                patient_val_loss, patient_dice_score = self._validation_step(patient_dict)
                
                epoch_val_loss += patient_val_loss
                epoch_val_dice += patient_dice_score
                
            epoch_val_loss /= len(self.val_volume_loader)
            epoch_val_dice /= len(self.val_volume_loader)

            # Clear CUDA cache
            torch.cuda.empty_cache()


            # Logging --
            logging.debug(f"Training loss: {epoch_train_loss}")
            logging.debug(f"Validation loss: {epoch_val_loss}")
            logging.debug(f"Validation dice: {epoch_val_dice}")
            logging.debug("")
            logging.debug("")

            if self.logging_config['enable-wandb']:
                wandb.log(
                          {
                           'train-loss': epoch_train_loss,
                           'val-loss': epoch_val_loss,
                           'val-dice': epoch_val_dice
                          },
                         step=epoch
                         )

            # Checkpointing --
            if self.training_config['enable-checkpointing']:
                if epoch % self.training_config['checkpoint-step'] == 0:
                    if self.input_data_config['in-bimodal']:    modality_str = 'petct'
                    else:    modality_str = self.input_modality.lower()
                    torch.save(self.model.state_dict(), 
                               f"{self.checkpoint_dir}/unet3d_{modality_str}_e{str(epoch).zfill(3)}.pt")


    def _train_step(self, batch_of_patches):

        # For bimodal input
        if self.input_data_config['is-bimodal']:
            # For PET and CT as separate volumes
            if self.input_data_config['input-representation'] == 'separate-volumes':
                PET_patches = batch_of_patches['PET'].to(self.device)
                CT_patches = batch_of_patches['CT'].to(self.device)
                # Pack these tensors into a list
                input_patches = [PET_patches, CT_patches]
            # For PET and CT as a single 2-channel volume
            if self.input_data_config['input-representation'] == 'multichannel-volume':
                input_patches = batch_of_patches['PET-CT'].to(self.device)
        
        # For unimodal input
        else: 
            modality = self.input_data_config['input-modality']
            input_patches = batch_of_patches[modality].to(self.device)

        # Target labelmap
        target_labelmap_patches = batch_of_patches['target-labelmap'].long().to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        pred_patches = self.model(input_patches)

        # Compute loss
        train_loss = self.criterion(pred_patches, target_labelmap_patches)

        # Generate loss gradients and back-propagate
        train_loss.backward()
        self.optimizer.step()

        return train_loss.item()


    def _validation_step(self, patient_dict):
        """

        """
        # Remove the batch dimension from input and target volumes of the patient dict
        for key, value in patient_dict.items():
            patient_dict[key] = value[0]

        # Get full list of patches
        patches_list = self.val_sampler.get_samples(patient_dict, num_patches=self.validation_config['valid-patches-per-volume'])

        # Stuff to accumulate
        patient_pred_patches_list = []
        patient_val_loss = 0  # To store average of losses over all patches 


        with torch.no_grad(): # Disable autograd
            # Take batch_of_patches_size number of patches at a time and push through the network
            for _ in range(0, self.validation_config['valid-patches-per-volume'], self.validation_config['batch-of-patches-size']):
                PET_patches = torch.stack([patches_list[i]['PET'] for i in range(self.validation_config['batch-of-patches-size'])], dim=0).to(self.device)
                target_labelmap_patches = torch.stack([patches_list[i]['target-labelmap'] for i in range(self.validation_config['batch-of-patches-size'])], dim=0).long().to(self.device)

                # Forward pass
                pred_patches = self.model(PET_patches)
                
                # Convert the predicted batch of probabilities to a list of labelmap patches, and store
                patient_pred_patches_list.extend(get_pred_labelmap_patches_list(pred_patches)) 

                # Compute validation loss
                val_loss = self.criterion(pred_patches, target_labelmap_patches)
                patient_val_loss += val_loss.item()
                
            # Calculate avergae validation loss for this patient
            patient_val_loss /= (self.validation_config['valid-patches-per-volume'] / self.validation_config['batch-of-patches-size'])

            # Aggregate and compute dice
            patient_pred_labelmap = self.val_aggregator.aggregate(patient_pred_patches_list, device=self.device) 
        patient_dice_score = volumetric_dice(patient_pred_labelmap.cpu().numpy(), patient_dict['target-labelmap'].cpu().numpy())

        return patient_val_loss, patient_dice_score