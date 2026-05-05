#!/bin/bash
#SBATCH --job-name=ootd_mask
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

RUN_NNODES=1

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/StableDiffusionExperimentss"
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup/dataset_ultimate_stratified_category}"
OUT_DIR="${OUT_DIR:-/iopsstor/scratch/cscs/dbartaula/experiments_assets}"

cd "${WORK_DIR}"

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
export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/cross-architecture:${PYTHONPATH:-}"
export WANDB_PROJECT=Stable_diffusion

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

for _if in hsn0 ib0 eth0 enp0s3 lo; do
  if ip -o link show "$_if" >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="$_if"
    export GLOO_SOCKET_IFNAME="$_if"
    break
  fi
done

# Derive a unique port from SLURM_JOB_ID to avoid collisions with other jobs.
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Compute master address BEFORE srun (scontrol is available on login/compute nodes).
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  SLURM_NNODES=$SLURM_NNODES  RUN_NNODES=$RUN_NNODES"

# Use SLURM_PROCID (set per-task by srun) for node_rank instead of SLURM_NODEID.
# Use the c10d rendezvous backend for robust multi-node coordination.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c '
  torchrun \
    --nnodes='"${RUN_NNODES}"' \
    --nproc_per_node=4 \
    --node_rank=${SLURM_PROCID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"${MASTER_ADDR}"':'"${MASTER_PORT}"' \
    --rdzv_id=${SLURM_JOB_ID} \
    cross-architecture/OOTDiffusion/train_ootdiffusion_mask_local.py --curvton_data_path '"${DATA_DIR}"' --category all --batch_size 8 --image_size 512 --num_workers 16 --max_steps 20000 --wandb_project Stable_diffusion --save_interval 1000 --image_log_interval 500 --output_dir '"${OUT_DIR}"' --run_name Stable_diffusion_train_ootdiffusion_mask
'
