#!/bin/bash
#SBATCH --job-name=ml_model_training
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/vhsingh/logs/ml_training-%j.log
#SBATCH --error=/home/vhsingh/logs/ml_training-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

PYTHON_SCRIPT="/home/vhsingh/Geo-UPSplus/CNO_Experiments/TrainFNO_time_L.py"  # <--- We'll use -m scripts.train below
# PROJECT_ROOT="/home/vhsingh/Geo-UPSplus/Autoregressive_Baseline_Scripts"
CHECKPOINT_FILE="/home/vhsingh/Geo-UPSplus/logs/checkpoint-${SLURM_JOB_ID}.txt"
export WANDB_API_KEY=3ff6a13421fb5921502235dde3f9a4700f33b5b8

if [[ -f $CHECKPOINT_FILE ]]; then
    echo "Resuming from checkpoint"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT_FILE"
else
    echo "Starting fresh"
    CHECKPOINT_ARG=""
fi

hostname
date
nvidia-smi
free -h

echo "Memory usage before running the script:"
python -c "import psutil; print(psutil.virtual_memory())"

# <--- Change directory into the project root
# cd "$PROJECT_ROOT"

# <--- Run your script as a module
python "$PYTHON_SCRIPT" $CHECKPOINT_ARG

echo "Job completed at $(date)"
