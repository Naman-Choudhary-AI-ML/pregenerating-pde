#!/bin/bash
#SBATCH --job-name=ml_model_training
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4           # Increase CPU cores
#SBATCH --mem=128G                  # Request 128 GB system memory
#SBATCH --output=/home/vhsingh/logs/ml_training-%j.log
#SBATCH --error=/home/vhsingh/logs/ml_training-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

PYTHON_SCRIPT="/home/vhsingh/Geo-UPSplus/FNO_Experiments/fno_openfoam_CE.py"
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

# Debug memory usage
echo "Memory usage before running the script:"
python -c "import psutil; print(psutil.virtual_memory())"

python $PYTHON_SCRIPT $CHECKPOINT_ARG

echo "Job completed at $(date)"
