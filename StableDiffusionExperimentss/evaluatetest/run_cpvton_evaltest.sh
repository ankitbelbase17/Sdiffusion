#!/bin/bash
set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/StableDiffusionExperimentss"
CURVTON_TEST_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
TRIPLET_TEST_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
STREET_TRYON_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
python cross-architecture/CPVTON/evaluate.py \
  --use_init_weights \
  --stage TOM \
  --curvton_test_data_path "$CURVTON_TEST_DIR" \
  --triplet_test_data_path "$TRIPLET_TEST_DIR" \
  --street_tryon_data_path "$STREET_TRYON_DIR" \
  --street_split validation \
  --batch_size 8 \
  --num_workers 8 \
  --eval_frac_curvton 0.02 \
  --eval_frac_triplet 0.02 \
  --eval_frac_street 0.02