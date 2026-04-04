#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --time=6:00:00
#SBATCH --output=/vol/bitbucket/%u/MEng-Individual-Project/logs/%x_%j.out
#SBATCH --error=/vol/bitbucket/%u/MEng-Individual-Project/logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-college-email>

LAMBDA=$1
DOMAIN=$2

echo "Starting run: lambda=$LAMBDA, domain=$DOMAIN"
echo "Job ID: $SLURM_JOB_ID"

# Redirect HF cache to bitbucket (cluster nodes can write here)
export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface

# Activate your venv
source /vol/bitbucket/$USER/MEng-Individual-Project/.venv/bin/activate

# Load CUDA
source /vol/cuda/12.0.0/setup.sh
nvidia-smi

# Go to your project
cd /vol/bitbucket/$USER/MEng-Individual-Project/

# Run training — pass lambda and domain as arguments
python src/mrt_train.py --domain $DOMAIN --lambda_val $LAMBDA "${@:3}"