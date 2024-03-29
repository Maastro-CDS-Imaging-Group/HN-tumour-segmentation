import yaml

from datautils.patch_sampling import get_num_valid_patches


def keys2kwargs(config_dict):
    """
    Eg: {'input-modality': 'PET'} --> {'input_modality': 'PET'}
    """
    kwargs = {}
    for key in config_dict.keys():
        kw = key.replace('-', '_')
        kwargs[kw] = config_dict[key]
    
    return kwargs


def build_config(cli_args, training=True):
    """
    Build a global config dict from the cli args
    """
    global_config = {}


    # Read YAML config files --
    with open(cli_args.data_config_file, 'r') as dc:
        yaml_data_config = yaml.safe_load(dc)
    with open(cli_args.nn_config_file, 'r') as nnc:
        yaml_nn_config = yaml.safe_load(nnc)
    if training:
        with open(cli_args.trainval_config_file, 'r') as tvc:
            yaml_trainval_config = yaml.safe_load(tvc)
    else:
        with open(cli_args.infer_config_file, 'r') as ic:
            yaml_infer_config = yaml.safe_load(ic)
    

    # Handle overrides (Modify the YAML derived config dicts) -- 
    if training:
        if cli_args.run_name is not None:
            yaml_trainval_config['trainer-kwargs']['logging_config']['run-name'] = cli_args.run_name


    # Get individual settings --    
    global_config['nn-name'] = yaml_nn_config['nn-name']


    # Construct kwargs dicts for the data pipeline --
    global_config['preprocessor-kwargs'] = yaml_data_config['preprocessor-kwargs']

    data_dir = f"{yaml_data_config['data-root-dir']}/{yaml_data_config['dataset-name'].split('-')[1]}_hecktor_nii"

    if training:

        train_dataset_kwargs = yaml_data_config['patient-dataset-kwargs'].copy()
        train_dataset_kwargs['data_dir'] = data_dir
        train_dataset_kwargs['patient_id_filepath'] = yaml_data_config['patient-id-filepath']
        train_dataset_kwargs['mode'] = yaml_trainval_config['trainer-kwargs']['training_config']['train-subset-name']        
        
        train_patch_sampler_kwargs = yaml_data_config['train-patch-sampler-kwargs']      
        train_patch_sampler_kwargs['patch_size'] = yaml_data_config['patch-size']
        train_patch_sampler_kwargs['volume_size'] = yaml_data_config['volume-size'] 

        train_patch_queue_kwargs = yaml_data_config['train-patch-queue-kwargs']

        train_patch_loader_kwargs = {'batch_size': yaml_data_config['batch-of-patches-size'], 
                                     'shuffle': False, 
                                     'num_workers': 0}
                                            
        val_dataset_kwargs = yaml_data_config['patient-dataset-kwargs'].copy()
        val_dataset_kwargs['augment_data'] = False
        val_dataset_kwargs['data_dir'] = data_dir
        val_dataset_kwargs['patient_id_filepath'] = yaml_data_config['patient-id-filepath']  
        val_dataset_kwargs['mode'] = yaml_trainval_config['trainer-kwargs']['validation_config']['val-subset-name'] 
               
        val_patch_sampler_kwargs = yaml_data_config['val-patch-sampler-kwargs']    
        val_patch_sampler_kwargs['patch_size'] = yaml_data_config['patch-size']    
        val_patch_sampler_kwargs['volume_size'] = yaml_data_config['volume-size'] 
        
        val_patch_aggregator_kwargs = yaml_data_config['val-patch-aggregator-kwargs']
        val_patch_aggregator_kwargs['patch_size'] = yaml_data_config['patch-size']
        val_patch_aggregator_kwargs['volume_size'] = yaml_data_config['volume-size']

        # Add into the global config
        global_config['train-dataset-kwargs'] = train_dataset_kwargs
        global_config['train-patch-sampler-kwargs'] = train_patch_sampler_kwargs
        global_config['train-patch-queue-kwargs'] = train_patch_queue_kwargs
        global_config['train-patch-loader-kwargs'] = train_patch_loader_kwargs
        global_config['val-dataset-kwargs'] = val_dataset_kwargs
        global_config['val-patch-sampler-kwargs'] = val_patch_sampler_kwargs
        global_config['val-patch-aggregator-kwargs'] = val_patch_aggregator_kwargs
        

    else: # For inference

        dataset_kwargs = yaml_data_config['patient-dataset-kwargs'].copy()
        dataset_kwargs['augment_data'] = False
        dataset_kwargs['data_dir'] = data_dir
        dataset_kwargs['patient_id_filepath'] = yaml_data_config['patient-id-filepath']  
        dataset_kwargs['mode'] = yaml_infer_config['inferer-kwargs']['inference_config']['subset-name']      
               
        patch_sampler_kwargs = yaml_data_config['val-patch-sampler-kwargs']   
        patch_sampler_kwargs['patch_size'] = yaml_data_config['patch-size']
        patch_sampler_kwargs['volume_size'] = yaml_data_config['volume-size']  

        patch_aggregator_kwargs = yaml_data_config['val-patch-aggregator-kwargs']
        patch_aggregator_kwargs['patch_size'] = yaml_data_config['patch-size']
        patch_aggregator_kwargs['volume_size'] = yaml_data_config['volume-size'] 
        
        # Add into the global config
        global_config['dataset-kwargs'] = dataset_kwargs
        global_config['patch-sampler-kwargs'] = patch_sampler_kwargs
        global_config['patch-aggregator-kwargs'] = patch_aggregator_kwargs
    

    # Integrate NN kwargs into global config --
    global_config['nn-kwargs'] = yaml_nn_config['nn-kwargs']


    # Construct the Trainer's or Inferer's kwargs  --
    val_valid_patches_per_volume = get_num_valid_patches(yaml_data_config['patch-size'], 
                                                     yaml_data_config['volume-size'], 
													 focal_point_stride=yaml_data_config['val-patch-sampler-kwargs']['focal_point_stride'],
													 padding=yaml_data_config['val-patch-sampler-kwargs']['padding'])

    input_data_config = {}
    input_data_config['is-bimodal'] = yaml_data_config['is-bimodal']

    if yaml_data_config['is-bimodal']: 
        input_data_config['input-modality'] = None
        input_data_config['input-representation'] = yaml_data_config['patient-dataset-kwargs']['input_representation']
    else: 
        input_data_config['input-modality'] = yaml_data_config['patient-dataset-kwargs']['input_modality']
        input_data_config['input-representation'] = None

    if training: # Trainer stuff

        hardware_config = yaml_trainval_config['trainer-kwargs']['hardware_config']

        training_config = yaml_trainval_config['trainer-kwargs']['training_config']
        training_config['dataset-name'] = yaml_data_config['dataset-name']

        validation_config = yaml_trainval_config['trainer-kwargs']['validation_config']
        validation_config['batch-of-patches-size'] = yaml_data_config['batch-of-patches-size']
        validation_config['valid-patches-per-volume'] = val_valid_patches_per_volume

        logging_config = yaml_trainval_config['trainer-kwargs']['logging_config']
        logging_config['wandb-config'] = {}
        logging_config['wandb-config']['patch-size'] = yaml_data_config['patch-size']

        # Add into the global config
        global_config['trainer-kwargs'] = {'hardware_config': hardware_config,
                                           'input_data_config': input_data_config,
                                           'training_config': training_config, 
                                           'validation_config': validation_config,
                                           'logging_config': logging_config}


    else: # Inferer stuff

        hardware_config = yaml_infer_config['inferer-kwargs']['hardware_config']
        inference_config = yaml_infer_config['inferer-kwargs']['inference_config']
        inference_config['dataset-name'] = yaml_data_config['dataset-name']
        inference_config['patient-id-filepath'] = yaml_data_config['patient-id-filepath']
        inference_config['batch-of-patches-size'] = yaml_data_config['batch-of-patches-size']
        inference_config['valid-patches-per-volume'] = val_valid_patches_per_volume

        # Add into the global config
        global_config['inferer-kwargs'] = {'hardware_config': hardware_config,
                                           'input_data_config': input_data_config,
                                           'inference_config': inference_config}

    
    return global_config