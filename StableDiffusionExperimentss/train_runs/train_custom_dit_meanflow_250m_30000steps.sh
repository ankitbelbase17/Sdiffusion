#!/bin/bash
#SBATCH --job-name=cdit_mf250_30000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/Sdiffusion/StableDiffusionExperimentss"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"
PHASE2_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1_1"

cd "$WORK_DIR"
unset PYTHONHOME || true
unset PYTHONPATH || true

CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch28_env_new}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV_NAME"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export WANDB_PROJECT=Stable_diffusion

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --standalone \
  --master_port=$MASTER_PORT \
  custom_model_pretraining/train_meanflow.py \
    --data_path ${DATA_DIR} \
    --phase2_data_path ${PHASE2_DIR} \
    --phase2_start_step 10000 \
    --curriculum none \
    --stage_steps 4000 \
    --max_steps 30000 \
    --batch_size 16 \
    --num_workers 16 \
    --image_size 512 \
    --patch_size 16 \
    --save_interval 1000 \
    --image_log_interval 250 \
    --gender all \
    --run_name Stable_diffusion_train_custom_dit_meanflow_250m_30000steps \
    --wandb_project Stable_diffusion
