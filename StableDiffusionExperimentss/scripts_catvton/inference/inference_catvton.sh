#!/bin/bash
#SBATCH --job-name=infer_catvton
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

usage() {
  cat <<EOF
Usage:
  sbatch scripts_catvton/inference/inference_catvton.sh \
    --checkpoint /abs/path/to/ckpt_final.pt \
    --person /abs/path/to/person.jpg \
    --cloth /abs/path/to/cloth.jpg \
    [--output results/tryon_catvton.png] \
    [--steps 50] \
    [--size 512] \
    [--device cuda] \
    [--base_model runwayml/stable-diffusion-v1-5] \
    [--ootd]
EOF
}

CHECKPOINT=""
PERSON_IMG=""
CLOTH_IMG=""
OUTPUT="results/tryon_catvton.png"
STEPS=50
SIZE=512
DEVICE="cuda"
BASE_MODEL="runwayml/stable-diffusion-v1-5"
OOTD_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --person) PERSON_IMG="$2"; shift 2 ;;
    --cloth) CLOTH_IMG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --base_model) BASE_MODEL="$2"; shift 2 ;;
    --ootd) OOTD_FLAG="--ootd"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$PERSON_IMG" || -z "$CLOTH_IMG" ]]; then
  echo "Error: --checkpoint, --person, and --cloth are required."
  usage
  exit 1
fi

python inference_catvton.py \
  --checkpoint "$CHECKPOINT" \
  --person "$PERSON_IMG" \
  --cloth "$CLOTH_IMG" \
  --output "$OUTPUT" \
  --steps "$STEPS" \
  --size "$SIZE" \
  --device "$DEVICE" \
  --base_model "$BASE_MODEL" \
  --fp16 \
  $OOTD_FLAG
