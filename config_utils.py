import yaml

from datautils.patch_sampling import get_num_valid_patches


def run_safety_checks():
    """
    Simple assertions
    """
    # TODO
    pass


def keys2kwargs(config_dict):
    """
    Eg: {'input-modality': 'PET'} --> {'input_modality' : 'PET'}
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


    # Read YAML config files
    data_config = yaml.safe_load(cli_args.data_config)
    nn_config = yaml.safe_load(cli_args.nn_config)
    if training:
        trainval_config = yaml.safe_load(cli_args.trainval_config)
    else:
        infer_config = yaml.safe_load(cli_args.infer_config)
    

    # Handle overrides
    if training:
        trainval_config['run-name'] = cli_args.run_name


    # Get individual settings
    val_valid_patches_per_volume = get_num_valid_patches(data_config['patch-size'], 
                                                     data_config['volume-size'], 
													 focal_point_stride=data_config['val-patch-sampler']['focal-point-stride'],
													 padding=data_config['val-patch-sampler']['padding'])
    if training:
        global_config['device'] = trainval_config['device']
    else:
        global_config['device'] = infer_config['device']


    # Construct kwargs dicts for the data pipeline
    preprocessor_kwargs = keys2kwargs(data_config['preprocessor'])

    if training:
        train_dataset_kwargs = keys2kwargs(data_config['patient-dataset'])
        train_dataset_kwargs['mode'] = trainval_config['training-config']['train-subset-name']
        train_patch_sampler_kwargs = keys2kwargs(data_config['train-patch-sampler'])      
        train_patch_queue_kwargs = keys2kwargs(data_config['train-patch-queue'])
    
        val_dataset_kwargs = keys2kwargs(data_config['patient-dataset'])
        val_dataset_kwargs['mode'] = trainval_config['validation-config']['val-subset-name']                
        val_patch_sampler_kwargs = data_config['val-patch-sampler']        
        val_patch_aggregator_kwargs = data_config['val-patch-aggregator']

        # Integrate into the global config
        global_config['train-dataset-kwargs'] = train_dataset_kwargs
        global_config['train-patch-sampler-kwargs'] = train_patch_sampler_kwargs
        global_config['train-patch-queue-kwargs'] = train_patch_queue_kwargs
        global_config['val-dataset-kwargs'] = val_dataset_kwargs
        global_config['val-patch-sampler-kwargs'] = val_patch_sampler_kwargs
        global_config['val-patch-aggregator-kwargs'] = val_patch_aggregator_kwargs


    else:
        # TODO
        pass

    
    # Integrate NN kwargs into global config 
    global_config['nn-kwargs': nn_config]


    # Construct the Trainer's or Inferer's kwargs
    input_data_config = {}
    input_data_config['in-bimodal'] = data_config['patient-dataset']

    if data_config['is-bimodal']: 
        input_data_config['input-representation'] = data_config['patient-dataset']['input-representation']
    else: 
        input_data_config['input-modality'] = data_config['patient-dataset']['input-modality']

    if training:
        training_config = trainval_config['training-config']
        training_config['dataset-name'] = data_config['dataset-name']

        validation_config = trainval_config['validation-config']
        validation_config['batch-of-patches-size'] = data_config['batch-of-patches-size']
        validation_config['valid-patches-per-volume']: val_valid_patches_per_volume

        logging_config = trainval_config['logging-config']
        logging_config['patch-size'] = data_config['patch-size']

        # Integrate into the global config
        global_config['trainer-kwargs'] = {'input_data_config': input_data_config,
                                           'training_config': training_config, 
                                           'validation_config': validation_config,
                                           'logging_config': logging_config}

    else:
        # TODO
        pass


    return global_config