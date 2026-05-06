#!/usr/bin/env python3
from evaluate_custom_dit_meanflow_250m_curvton_splits import main as _main, COMMAND

COMMAND[:] = [
    "python", "custom_model_pretraining/evaluate_fid_kid.py",
    "--approach", "meanflow",
    "--checkpoint", "__CKPT_PATH__",
    "--curvton_test_data_path", "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test",
    "--triplet_test_data_path", "",
    "--street_tryon_data_path", "",
    "--image_size", "512",
    "--patch_size", "16",
    "--hidden_size", "1280",
    "--depth", "9",
    "--num_heads", "20",
    "--batch_size", "16",
    "--num_workers", "8",
    "--eval_frac_curvton", "0.80",
    "--eval_frac_curvton_overall", "0.25",
    "--eval_frac_curvton_extra", "0.25",
    "--output_json", "/iopsstor/scratch/cscs/dbartaula/experiments_assets/Stable_diffusion_train_custom_dit_meanflow_250m_30000steps/eval_curvton_overall_extra.json",
]

if __name__ == "__main__":
    raise SystemExit(_main())

