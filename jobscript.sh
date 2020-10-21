#!/usr/local_rwth/bin/zsh


# Job configuration ---

#SBATCH --job-name=test2
#SBATCH --output=slurm_job_logs/test2.%j.log

## OpenMP settings
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

## Request for a node with 2 Tesla P100 GPUs
#SBATCH --gres=gpu:pascal:2

#SBATCH --time=15:00:00

## TO use the UM DKE project account
# #SBATCH --account=um_dke


# Load CUDA 
module load cuda

# Debug info
echo; echo
nvidia-smi
echo; echo

# Execute script
python_interpreter="../../maastro_env/bin/python3"
python_file="./train.py"

$python_interpreter $python_file 


#------------------------
# Note: All relative paths are relative to the directory containing the job script