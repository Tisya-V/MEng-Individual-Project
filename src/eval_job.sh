#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/%u/MEng-Individual-Project/logs/%x_%j.out
#SBATCH --error=/vol/bitbucket/%u/MEng-Individual-Project/logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email>

DOMAIN=$1
LAM=$2

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface
export MPLCONFIGDIR=/vol/bitbucket/$USER/.cache/matplotlib
export FONTCONFIG_PATH=/vol/bitbucket/$USER/.cache/fontconfig

source /vol/cuda/12.0.0/setup.sh
source /vol/bitbucket/$USER/MEng-Individual-Project/.venv/bin/activate

nvidia-smi

cd /vol/bitbucket/$USER/MEng-Individual-Project/
python src/mrt_eval.py --domain $DOMAIN --lam $LAM