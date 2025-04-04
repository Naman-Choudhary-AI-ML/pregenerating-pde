#!/bin/bash
#SBATCH --job-name=ml_model_training
#SBATCH --partition=general         # Use the general partition
#SBATCH --ntasks=1                  # Number of tasks (1 for multi-GPU on one node)
#SBATCH --gpus=1                    # Request 1 GPUs
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --time=48:00:00
#SBATCH --mem=128G                  # Request 128 GB system memory
#SBATCH --output=/home/vhsingh/logs/ml_training-%j.log  # Log file
#SBATCH --error=/home/vhsingh/logs/ml_training-%j.err   # Error file
#SBATCH --mail-type=END,FAIL        # Email on job completion or failure
#SBATCH --mail-user=vhsingh@andrew.cmu.edu  # Your email
#SBATCH --signal=B:USR1@60          # Signal for requeueing
#SBATCH --requeue                   # Automatically requeue if preempted

# --- Environment Variables ---
export WANDB_API_KEY=3ff6a13421fb5921502235dde3f9a4700f33b5b8  # Set your WandB API key

# --- Set the parent directory where checkpoints are saved ---
CHECKPOINT_DIR="/data/user_data/vhsingh/checkpoints/Research"

# --- Identify the Latest Checkpoint Subfolder ---
# This command lists all subfolders named checkpoint-*, sorts them by numeric value, and picks the last one.
LATEST_CHECKPOINT_DIR=$(ls -d -v "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | tail -n 1)

if [[ -d "$LATEST_CHECKPOINT_DIR" ]]; then
    echo "Resuming from checkpoint directory: $LATEST_CHECKPOINT_DIR"
    # The script likely expects --checkpoint (for loading) 
    # while --checkpoint_path is where new checkpoints get saved
    CHECKPOINT_ARG="--checkpoint $LATEST_CHECKPOINT_DIR"
else
    echo "No checkpoint subfolder found, starting fresh"
    CHECKPOINT_ARG=""
fi

# --- System Information ---
hostname
date
nvidia-smi
free -h
echo "Memory usage before running the script:"

# --- Run the Training Command ---
accelerate launch train.py \
    --config /home/vhsingh/Geo-UPSplus/Scot_Swin_Experiments/configs/run.yaml \
    --wandb_run_name "Hole_Location_scot_FPO_Pipe-400-100-80-nofinetuning" \
    --wandb_project_name "Research" \
    --data_path /home/vhsingh/test/scaled_NS_Regular.npy \
    --checkpoint_path $CHECKPOINT_DIR \
    $CHECKPOINT_ARG \
    2>&1 | tee /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log

# --- Capture Exit Status ---
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Job FAILED with exit status $exit_status" | tee -a /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log
else
    echo "Job COMPLETED SUCCESSFULLY" | tee -a /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log
fi