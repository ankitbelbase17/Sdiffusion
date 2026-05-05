#!/bin/bash
#SBATCH --job-name=none_cat_24000_fullres
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Usage:
#   sbatch train_no_curriculum_catvton_24000steps_fullres.sh --run_nodes 1
#   sbatch train_no_curriculum_catvton_24000steps_fullres.sh --run_nodes 2
RUN_NNODES="${SLURM_NNODES:-2}"
if [[ "${1:-}" == "--run_nodes" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "ERROR: --run_nodes requires a value (1 or 2)" >&2
    exit 1
  fi
  RUN_NNODES="$2"
  shift 2
fi
if [[ "$RUN_NNODES" != "1" && "$RUN_NNODES" != "2" ]]; then
  echo "ERROR: run_nodes must be 1 or 2 (got: $RUN_NNODES)" >&2
  exit 1
fi
if (( RUN_NNODES > SLURM_NNODES )); then
  echo "ERROR: requested run_nodes=$RUN_NNODES but allocation has SLURM_NNODES=$SLURM_NNODES" >&2
  exit 1
fi
export RUN_NNODES

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/StableDiffusionExperimentss"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"

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
srun --nodes=${RUN_NNODES} --ntasks=${RUN_NNODES} --ntasks-per-node=1 bash -c '
  torchrun \
    --nnodes='"${RUN_NNODES}"' \
    --nproc_per_node=4 \
    --node_rank=${SLURM_PROCID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"${MASTER_ADDR}"':'"${MASTER_PORT}"' \
    --rdzv_id=${SLURM_JOB_ID} \
    train.py --dataset curvton --curriculum none --stage_steps 9600 --max_steps 24000 --curvton_data_path '"${DATA_DIR}"' --batch_size 4 --num_workers 16 --gender all --image_size 0 --save_interval 1000 --image_log_interval 500 --skip_eval --run_name Stable_diffusion_train_no_curriculum_catvton_24000steps_fullres
'
