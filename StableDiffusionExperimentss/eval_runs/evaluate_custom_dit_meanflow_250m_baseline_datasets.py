#!/usr/bin/env python3
from evaluate_custom_dit_meanflow_250m_curvton_splits import main as _main, COMMAND

COMMAND[:] = [
    "python", "custom_model_pretraining/evaluate_fid_kid.py",
    "--approach", "meanflow",
    "--checkpoint", "__CKPT_PATH__",
    "--curvton_test_data_path", "",
    "--triplet_test_data_path", "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1",
    "--street_tryon_data_path", "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon",
    "--image_size", "512",
    "--patch_size", "16",
    "--hidden_size", "1280",
    "--depth", "9",
    "--num_heads", "20",
    "--batch_size", "16",
    "--num_workers", "8",
    "--eval_frac_triplet", "0.25",
    "--eval_frac_street", "0.25",
    "--output_json", "/iopsstor/scratch/cscs/dbartaula/experiments_assets/Stable_diffusion_train_custom_dit_meanflow_250m_30000steps/eval_baseline_datasets.json",
]

if __name__ == "__main__":
    raise SystemExit(_main())

