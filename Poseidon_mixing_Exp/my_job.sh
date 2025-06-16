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

easy_counts=(200 150 120  80  50)     # no‑hole / “easy”
hard_counts=(  1  99 179 319 949)     # hole / “hard”

easy=${easy_counts[$SLURM_ARRAY_TASK_ID]}
hard=${hard_counts[$SLURM_ARRAY_TASK_ID]}
total=$((easy + hard))                # still useful for logging
alpha=$(awk "BEGIN {print $hard / $total}")

# ── 2) run / checkpoint names now mention easy+hard instead of α ──────────────
export WANDB_RUN_NAME="mix_${easy}E_${hard}H_task${SLURM_ARRAY_TASK_ID}"
export WANDB_MODE=offline

CHECKPOINT_BASE="/data/user_data/vhsingh/checkpoints/Research"
JOB_CKPT_DIR="${CHECKPOINT_BASE}/${WANDB_RUN_NAME}"
mkdir -p "$JOB_CKPT_DIR"

# … resume‑logic unchanged …

# ── 3) launch the container ----------------------------------------------------
apptainer exec --unsquash --nv \
  --bind $PWD:$PWD \
  --bind $JOB_CKPT_DIR:$JOB_CKPT_DIR \
  … other binds … \
  my_experiment.sif \
  python /home/vhsingh/Poseidon_mixing_Exp/scOT/mixingexp.py \
      --config            /home/vhsingh/Poseidon_mixing_Exp/configs/mixing.yaml \
      --num_easy          "$easy" \
      --num_hard          "$hard" \
      --wandb_run_name    "$WANDB_RUN_NAME" \
      --wandb_project_name Research \
      --checkpoint_path   "$JOB_CKPT_DIR" \
      $CHECKPOINT_ARG \
      --data_path         /home/vhsingh/.../NS_LDC_reg-001.npy \
      --finetune_from     camlab-ethz/Poseidon-T \
      --replace_embedding_recovery \
2>&1 | tee /home/vhsingh/logs/ml_training-$SLURM_JOB_ID-detailed.log
