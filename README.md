# Head-and-Neck GTV Segmentation in Multimodal FDG-PET/CT scans

Uses code from [this repo](https://github.com/Maastro-CDS-Imaging-Group/PET-CT-data-pipeline) for PET-CT data processing, patch based data loading, and patch aggregation



## Directory structures for benchmarking experiment

- Basic directory structure for *saved_models*, *model_predictions* and *model_performances*:
    ```
    root_dir
        |
        |- hecktor-crFHN_rs113
        |   |
        |   |- msam3d_petct
        |
        |- hecktor-crS_rs113
            |
            |- msam3d_petct
            |- unet3d_petct
            |- unet3d_pet
            |- unet3d_ct
    ```

- Cross-validation sub-directories:

    - For *saved_models* and *model_predictions*, each model directory (ex. unet3d_petct) additionally contains 4 sub-dirs for the crossval splits, like so:
        ```
        unet3d_petct
            |
            |- crossval-CHGJ
            |- crossval-CHMR
            |- crossval-CHUM
            |- crossval-CHUS
        ```
        
    - In case of *model_performances*, the scorecard is computed for each model across all the crossval splits. Hence, this root directory doesn't have the crossval sub-dirs


- Late fusion labelmaps and scorecard: Additional model directory named *unet3d_latefusion* within the *hecktor-crS_rs113* directory, in case of *model_predictions* and *model_performances*
