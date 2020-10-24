import logging
import numpy as np
import torch
from tqdm import tqdm
import wandb

from trainutils.loss_functions import build_loss_function
from datautils.patch_aggregation import PatchAggregator3D, get_pred_labelmap_patches_list
from trainutils.metrics import volumetric_dice

logging.basicConfig(level=logging.DEBUG)



class Trainer():
    def __init__(self, 
                model, 
                train_patch_loader, val_volume_loader, val_sampler, val_aggregator,
                device,
                input_data_config,
                training_config, 
                validation_config, 
                logging_config):
        
        self.model = model
        self.train_patch_loader = train_patch_loader
        self.val_volume_loader = val_volume_loader
        self.val_sampler = val_sampler
        self.val_aggregator = val_aggregator

        self.device = device 

        # Input data config TODO: Incorporate into training and validation steps
        self.input_is_bimodal = input_data_config['is-bimodal']
        self.input_modality = input_data_config['input-modality']
        self.input_representation = input_data_config['input-representation']

        # Training config
        self.criterion = build_loss_function(training_config['loss-name'], self.device)
        self.num_epochs = training_config['num-epochs']
        self.learning_rate = training_config['learning-rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.learning_rate)
        
        self.start_epoch = 1
        self.enable_checkpointing = training_config['enable-checkpointing']
        self.checkpoint_step = training_config['checkpoint-step']
        self.checkpoint_dir = training_config['checkpoint-dir']

        self.continue_from_checkpoint = training_config['continue-from-checkpoint']
        if self.continue_from_checkpoint:
            checkpoint_filename = training_config['checkpoint-filename']
            logging.debug(f"Loading checkpoint: {checkpoint_filename}")
            # Checkpoint name example - unet3d_pet_e005.pt
            self.model.load_state_dict(torch.load(f"{self.checkpoint_dir}/{checkpoint_filename}"))
            self.start_epoch = int(checkpoint_filename.split('.')[0][-3:]) + 1
            logging.debug(f"Continuing from epoch {self.start_epoch}")

        # Validation config
        self.batch_of_patches_size = validation_config['batch-of-patches-size']
        self.valid_patches_per_volume = validation_config['valid-patches-per-volume']

        # Logging config
        self.enable_wandb = logging_config['enable-wandb']
        if self.enable_wandb:
            wandb_entity = logging_config['wandb-entity']
            wandb_project = logging_config['wandb-project']
            wandb_run_name = logging_config['wandb-run-name']
            wandb_config = logging_config['wandb-config']

            wandb.init(entity=wandb_entity,
			           project=wandb_project,
			           name=wandb_run_name,
			           config=wandb_config
			          )
            wandb.config.update({'batch_of_patches_size': self.batch_of_patches_size,
                                'learning_rate': self.learning_rate,
                                'start_epoch': self.start_epoch,
                                'num_epochs': self.start_epoch + self.num_epochs})
            wandb.watch(model)


    def train(self):
        """
        Training loop
        """
        
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
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
                patient_val_loss, dice_score = self._validation_step(patient_dict)
                
                epoch_val_loss += patient_val_loss
                epoch_val_dice += dice_score
                
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

            if self.enable_wandb:
                wandb.log(
                          {
                           'train-loss': epoch_train_loss,
                           'val-loss': epoch_val_loss,
                           'val-dice': epoch_val_dice
                          },
                         step=epoch
                         )

            # Checkpointing --
            if self.enable_checkpointing:
                if epoch % self.checkpoint_step == 0:
                    if self.input_is_bimodal:    modality_str = 'petct'
                    else:    modality_str = self.input_modality.lower()
                    torch.save(self.model.state_dict(), 
                               f"{self.checkpoint_dir}/unet3d_{modality_str}_e{str(epoch).zfill(3)}.pt")


    def _train_step(self, batch_of_patches):

        PET_patches = batch_of_patches['PET'].to(self.device)
        GTV_labelmap_patches = batch_of_patches['GTV-labelmap'].long().to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        pred_patches = self.model(PET_patches)

        # Compute loss
        train_loss = self.criterion(pred_patches, GTV_labelmap_patches)

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
        patches_list = self.val_sampler.get_samples(patient_dict, num_patches=self.valid_patches_per_volume)

        # Stuff to accumulate
        patient_pred_patches_list = []
        patient_val_loss = 0  # To store average of losses over all patches 


        with torch.no_grad(): # Disable autograd
            # Take batch_of_patches_size number of patches at a time and push through the network
            for _ in range(0, self.valid_patches_per_volume, self.batch_of_patches_size):
                PET_patches = torch.stack([patches_list[i]['PET'] for i in range(self.batch_of_patches_size)], dim=0).to(self.device)
                GTV_labelmap_patches = torch.stack([patches_list[i]['GTV-labelmap'] for i in range(self.batch_of_patches_size)], dim=0).long().to(self.device)

                # Forward pass
                pred_patches = self.model(PET_patches)
                
                # Convert the predicted batch of probabilities to a list of labelmap patches, and store
                patient_pred_patches_list.extend(get_pred_labelmap_patches_list(pred_patches)) 

                # Compute validation loss
                val_loss = self.criterion(pred_patches, GTV_labelmap_patches)
                patient_val_loss += val_loss.item()
                
            # Calculate avergae validation loss for this patient
            patient_val_loss /= (self.valid_patches_per_volume / self.batch_of_patches_size)

            # Aggregate and compute dice
            pred_labelmap_volume = self.val_aggregator.aggregate(patient_pred_patches_list, device=self.device) 
        dice_score = volumetric_dice(pred_labelmap_volume.cpu().numpy(), patient_dict['GTV-labelmap'].cpu().numpy())

        return patient_val_loss, dice_score