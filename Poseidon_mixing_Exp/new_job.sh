#!/usr/bin/env bash
#SBATCH --job-name=mix-conda
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.log   # one combined file per array-task
#SBATCH --error=logs/%x_%A_%a.log    # (stderr → same file)
## OPTIONAL:  e-mail yourself if it fails
## #SBATCH --mail-type=FAIL --mail-user=you@cmu.edu

# ── 0) ultra-strict bash + live tee to another file ───────────────────────────
set -Eeuo pipefail
LOGDIR=$HOME/logs
mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_$(date +%s).log"
exec > >(tee -i "$RUNLOG") 2>&1          # duplicate everything to $RUNLOG
PS4='+ ${BASH_SOURCE}:${LINENO}: '

trap 'echo "❌ ERROR on line $LINENO (exit $?)" >&2' ERR

# turn on command echoing **after** PS4 is defined
set -x

# ── 1) load modules & activate conda env ──────────────────────────────────────
# module purge
# module load cuda/12.1
# module load anaconda3/2023.07

# if ! command -v conda &>/dev/null; then
#   source ~/miniconda3/etc/profile.d/conda.sh   # CHANGE path if miniconda is elsewhere
# fi

# conda activate poseidon        # CHANGE to your env name

echo "[INFO] Using Python: $(which python)"
python -V

# ── 2) experiment-specific parameters ----------------------------------------
easy_counts=(824 823 816 783 742 618 412 124)
hard_counts=( 0 1 8   41  82  206 412 700)

EASY=${easy_counts[$SLURM_ARRAY_TASK_ID]}
HARD=${hard_counts[$SLURM_ARRAY_TASK_ID]}

export WANDB_RUN_NAME="mix_${EASY}E_${HARD}H_task${SLURM_ARRAY_TASK_ID}"
export WANDB_MODE=offline

CKPT_BASE=/data/user_data/vhsingh/checkpoints/Research
CKPT_DIR="${CKPT_BASE}/${WANDB_RUN_NAME}"
mkdir -p "$CKPT_DIR"

LATEST=$(ls -d -v "$CKPT_DIR"/checkpoint-* 2>/dev/null | tail -n1 || true)
[[ -d "$LATEST" ]] && CHECKPOINT_ARG="--checkpoint $LATEST" || CHECKPOINT_ARG=""

# ── 3) echo everything we decided, for posterity ------------------------------
echo "[INFO] Slurm job        : $SLURM_JOB_ID / task $SLURM_ARRAY_TASK_ID"
echo "[INFO] Node             : $(hostname)"
echo "[INFO] EASY / HARD      : $EASY  /  $HARD"
echo "[INFO] Checkpoint dir   : $CKPT_DIR"
echo "[INFO] Resume checkpoint: ${CHECKPOINT_ARG:-none}"

# ── 4) paths to code & data ---------------------------------------------------
CODE_DIR=/home/vhsingh/Geo-UPSplus/Poseidon_mixing_Exp
PY_MAIN=${CODE_DIR}/scOT/mixingexp.py
CONFIG=${CODE_DIR}/configs/mixing.yaml
DATA=/data/group_data/sage_lab_complex_geometry/FPO_Cylinder_Multiple_Holes_6400_Normalised.npy

# verify they exist *before* wasting any GPU time
for p in "$PY_MAIN" "$CONFIG" "$DATA"; do
  [[ -e "$p" ]] || { echo "[FATAL] Path does not exist: $p"; exit 3; }
done

# ── 5) launch training --------------------------------------------------------
python "$PY_MAIN" \
  --config "$CONFIG" \
  --num_easy "$EASY" \
  --num_hard "$HARD" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_project_name Research \
  --checkpoint_path "$CKPT_DIR" \
  $CHECKPOINT_ARG \
  --data_path "$DATA" \
  --finetune_from camlab-ethz/Poseidon-T \
  --replace_embedding_recovery
