#!/usr/bin/env bash
#SBATCH --job-name=alpha-sweep
#SBATCH --output=logs/alpha_%a.out
#SBATCH --error=logs/alpha_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --array=0-4

mkdir -p /home/vhsingh/logs

# 1) pick your hyperparameter
alphas=(0.995 0.803 0.668 0.501 0.2002)
totals=(201    249    299  399   999)

alpha=${alphas[$SLURM_ARRAY_TASK_ID]}
total=${totals[$SLURM_ARRAY_TASK_ID]}

# 2) define run name
export WANDB_RUN_NAME="mixing_alpha_${alpha}_task${SLURM_ARRAY_TASK_ID}"
export WANDB_MODE=offline

# 3) base checkpoint dir + per-job subfolder
CHECKPOINT_BASE="/data/user_data/vhsingh/checkpoints/Research"
JOB_CKPT_DIR="${CHECKPOINT_BASE}/${WANDB_RUN_NAME}"
mkdir -p "$JOB_CKPT_DIR"

# 4) check for resume
LATEST=$(ls -d -v "${JOB_CKPT_DIR}"/checkpoint-* 2>/dev/null | tail -n 1)
if [[ -d "$LATEST" ]]; then
  echo "Resuming from $LATEST"
  CHECKPOINT_ARG="--checkpoint $LATEST"
else
  echo "No checkpoint found in $JOB_CKPT_DIR, starting fresh"
  CHECKPOINT_ARG=""
fi

# --- System info ---
hostname; date; nvidia-smi; free -h; echo "Memory usage before run:"

# 5) launch, binding the exact JOB_CKPT_DIR
apptainer exec --unsquash --nv \
  --bind $PWD:$PWD \
  --bind $JOB_CKPT_DIR:$JOB_CKPT_DIR \
  --bind $HOME/Poseidon_mixing_Exp/scOT:/workspace/scOT \
  --bind /data:/data \
  --pwd $PWD \
  my_experiment.sif \
  python /home/vhsingh/Poseidon_mixing_Exp/scOT/mixingexp.py \
    --config           /home/vhsingh/Poseidon_mixing_Exp/configs/mixing.yaml \
    --alpha            "$alpha" \
    --total_trajectories "$total" \
    --wandb_run_name   "$WANDB_RUN_NAME" \
    --wandb_project_name Research \
    --checkpoint_path  "$JOB_CKPT_DIR" \
    $CHECKPOINT_ARG \
    --data_path        /home/vhsingh/.../NS_LDC_reg-001.npy \
    --finetune_from    camlab-ethz/Poseidon-T \
    --replace_embedding_recovery \
2>&1 | tee /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log

exit_status=${PIPESTATUS[0]}
if [ $exit_status -ne 0 ]; then
  echo "Job FAILED with exit status $exit_status" | tee -a /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log
else
  echo "Job COMPLETED SUCCESSFULLY" | tee -a /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log
fi
