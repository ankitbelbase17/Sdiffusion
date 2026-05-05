#!/bin/bash
#SBATCH --job-name=infer_hunyuan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/StableDiffusionExperimentss"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# ── Edit these paths ──────────────────────────────────────────
CHECKPOINT="/path/to/ckpt_final.pt"
PERSON_IMG="/path/to/person.jpg"
CLOTH_IMG="/path/to/cloth.jpg"
OUTPUT="results/tryon_hunyuan.png"
# ──────────────────────────────────────────────────────────────

python inference_hunyuan.py \
  --checkpoint "$CHECKPOINT" \
  --person "$PERSON_IMG" \
  --cloth "$CLOTH_IMG" \
  --output "$OUTPUT" \
  --steps 50 \
  --fp16

