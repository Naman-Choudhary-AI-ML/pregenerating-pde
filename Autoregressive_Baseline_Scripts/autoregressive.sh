#!/usr/bin/env bash
#SBATCH --job-name=mix-conda
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log

set -Eeuo pipefail
LOGDIR=$HOME/logs
mkdir -p "$LOGDIR"
exec > >(tee -i "$LOGDIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_$(date +%s).log") 2>&1
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

###############################################################################
# 0) tensorboard / wandb etc.
###############################################################################
export WANDB_MODE=online
export WANDB_API_KEY=3ff6a13421fb5921502235dde3f9a4700f33b5b8

###############################################################################
# 1) easy / hard counts  (pick whatever grid you like)
###############################################################################
easy_counts=(1 49 99 199 399 799 1599)
hard_counts=(200 200 200 200 200 200 200)

EASY=${easy_counts[$SLURM_ARRAY_TASK_ID]}
HARD=${hard_counts[$SLURM_ARRAY_TASK_ID]}

###############################################################################
# 2) paths
###############################################################################
CODE=/home/vhsingh/Geo-UPSplus/Autoregressive_Baseline_Scripts
MAIN=${CODE}/scripts/train.py              # <—— has the new CLI flags
CONF=${CODE}/config/config.yaml

NO_HOLE=/data/group_data/sage_lab_complex_geometry/new_FPO_dataset/FPO_Cylinder_Hole_Location_3000.npy               # easy
HOLE=/data/group_data/sage_lab_complex_geometry/FPO_Cylinder_Multiple_Holes_3200_Normalised.npy                   # hard

###############################################################################
# 3) checkpoint directory per run
###############################################################################
CKPT=/data/user_data/vhsingh/checkpoints/mix_${EASY}E_${HARD}H
mkdir -p "$CKPT"
LATEST=$(ls -d -v "$CKPT"/checkpoint-* 2>/dev/null | tail -n1 || true)
[[ -d "$LATEST" ]] && RESUME="--checkpoint $LATEST" || RESUME=""

###############################################################################
# 4) launch
###############################################################################
python -m scripts.train \
  --easy-train "$EASY" \
  --hard-train "$HARD" \
  --data_path "$NO_HOLE" "$HOLE" \
  --config "$CONF" \
  --checkpoint_root "$CKPT" \
  $RESUME
