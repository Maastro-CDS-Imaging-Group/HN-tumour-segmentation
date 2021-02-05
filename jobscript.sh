#!/usr/local_rwth/bin/zsh


# Job configuration ---

#SBATCH --job-name=crFHN-msam3d-petct-cvCHUM-gtvweighted-histfix
#SBATCH --output=slurm_job_logs/crFHN-msam3d-petct-cvCHUM-gtvweighted-histfix.%j.log

## OpenMP settings
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

## Request for a node with 2 Tesla P100 GPUs
#SBATCH --gres=gpu:pascal:2

#SBATCH --time=120:00:00

## TO use the UM DKE project account
# #SBATCH --account=um_dke


# Load CUDA 
module load cuda

# Debug info
echo; echo
nvidia-smi
echo; echo

# Execute training script
python_interpreter="../../maastro_env/bin/python3"
python_file="./training_script.py"

data_config_file="./config_files/data-crFHN_rs113-petct_default.yaml"
nn_config_file="./config_files/nn-msam3d_default.yaml"
trainval_config_file="./config_files/trainval-default.yaml"

run_name="crFHN-msam3d-petct-cvCHUM-gtvweighted-histfix"

$python_interpreter $python_file --data_config_file $data_config_file \
                                 --nn_config_file $nn_config_file \
                                 --trainval_config_file $trainval_config_file \
                                 --run_name $run_name 


#------------------------
# Note: All relative paths are relative to the directory containing the job script
