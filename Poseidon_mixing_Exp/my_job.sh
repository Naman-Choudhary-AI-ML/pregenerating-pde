#!/usr/bin/env bash
#SBATCH --job-name=mix-sweep
#SBATCH --output=logs/mix_%a.out
#SBATCH --error=logs/mix_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --array=0-7

set -euo pipefail                       # bail fast on missing vars / errors

# ── 0) constants ──────────────────────────────────────────────────────────────
IMAGE=/home/vhsingh/Geo-UPSplus/Poseidon_mixing_Exp/my_experiment.sif
CODE_DIR=/home/vhsingh/Poseidon_mixing_Exp      # where mixingexp.py lives
LOG_DIR=/home/vhsingh/logs
CHECKPOINT_BASE=/data/user_data/vhsingh/checkpoints/Research

mkdir -p "$LOG_DIR"

# ── 1) choose split ───────────────────────────────────────────────────────────
easy_counts=(100 100 100 100 100 100 100 100)
hard_counts=(  1   5  10  50 100 200 400 800)

easy=${easy_counts[$SLURM_ARRAY_TASK_ID]}
hard=${hard_counts[$SLURM_ARRAY_TASK_ID]}

export WANDB_RUN_NAME="mix_${easy}E_${hard}H_task${SLURM_ARRAY_TASK_ID}"
export WANDB_MODE=offline

JOB_CKPT_DIR="${CHECKPOINT_BASE}/${WANDB_RUN_NAME}"
mkdir -p "$JOB_CKPT_DIR"

# ── 2) resume logic (unchanged) ───────────────────────────────────────────────
LATEST=$(ls -d -v "${JOB_CKPT_DIR}"/checkpoint-* 2>/dev/null | tail -n 1)
[[ -d "$LATEST" ]] && CHECKPOINT_ARG="--checkpoint $LATEST" || CHECKPOINT_ARG=""

# ── 3) sanity-checks before launching container ──────────────────────────────
if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: container image not found: $IMAGE" >&2; exit 1
fi
for d in "$CODE_DIR" "$JOB_CKPT_DIR"; do
  [[ -d "$d" ]] || { echo "ERROR: bind-mount target missing: $d" >&2; exit 1; }
done

# ── 4) launch ────────────────────────────────────────────────────────────────
apptainer exec --unsquash --nv \
  --bind "$PWD":"$PWD" \
  --bind "$JOB_CKPT_DIR":"$JOB_CKPT_DIR" \
  --bind "$CODE_DIR":"$CODE_DIR" \
  --bind /data:/data \
  "$IMAGE" \
  python "$CODE_DIR/scOT/mixingexp.py" \
    --config "$CODE_DIR/configs/mixing.yaml" \
    --num_easy "$easy" \
    --num_hard "$hard" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_project_name Research \
    --checkpoint_path "$JOB_CKPT_DIR" \
    $CHECKPOINT_ARG \
    --data_path /data/group_data/sage_lab_complex_geometry/FPO_Cylinder_Multiple_Holes_6400_Normalised.npy \
    --finetune_from camlab-ethz/Poseidon-T \
    --replace_embedding_recovery \
2>&1 | tee "$LOG_DIR/ml_training-$SLURM_JOB_ID.log"
