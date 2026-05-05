#!/bin/bash
#SBATCH --job-name=idm_vton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# IDM-VTON local implementation aligned to the official architecture/loss.
# Repo: https://github.com/yisol/IDM-VTON
# Architecture: SDXL inpainting-style UNet with garment conditioning.
# Loss: diffusion denoising objective on predicted noise (epsilon), with scheduler weighting.

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/StableDiffusionExperimentss"
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate}"
OUT_DIR="${OUT_DIR:-/iopsstor/scratch/cscs/dbartaula/experiments_assets}"

cd "${WORK_DIR}"

# Clean inherited Python env vars first; bad PYTHONHOME/PYTHONPATH can break stdlib encodings.
unset PYTHONHOME || true
unset PYTHONPATH || true

# Activate the intended conda env used on the cluster.
CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch28_env_new}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV_NAME"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/cross-architecture:${PYTHONPATH:-}"
export WANDB_PROJECT=Stable_diffusion

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

# Select a concrete network interface name for Gloo/NCCL ("hsn" alone is not a valid device).
for _if in hsn0 ib0 eth0 enp0s3 lo; do
  if ip -o link show "$_if" >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="$_if"
    export GLOO_SOCKET_IFNAME="$_if"
    break
  fi
done

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "[DEBUG] Host=$(hostname) JobID=${SLURM_JOB_ID:-unknown}"
echo "[DEBUG] Conda env=$CONDA_ENV_NAME CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "[DEBUG] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "[DEBUG] Python=$(which python || true) Torchrun=$(which torchrun || true)"
python -V || true

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --standalone \
  --master_port=$MASTER_PORT \
  cross-architecture/IDMVTON/train_idm_vton_local.py \
  --curvton_data_path "${DATA_DIR}" \
  --batch_size 6 \
  --image_size 0 \
  --num_workers 16 \
  --max_steps 28000 \
  --save_interval 1000 \
  --output_dir "${OUT_DIR}" \
  --no_resume \
  --run_name Stable_diffusion_train_idm_vton
