#!/usr/bin/env python3
from evaluate_ootdiffusion_curvton import main as _main, COMMAND

COMMAND[:] = [
    "python",
    "cross-architecture/OOTDiffusion/evaluate.py",
    "--checkpoint",
    "__CKPT_PATH__",
    "--curvton_test_data_path",
    "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test",
    "--triplet_test_data_path",
    "",
    "--street_tryon_data_path",
    "",
    "--batch_size",
    "16",
    "--num_workers",
    "8",
    "--num_inference_steps",
    "30",
    "--curvton_splits",
    "traditional,non_traditional,dresses,upper_body,lower_body",
    "--eval_frac_curvton_extra", "0.25",
    "--output_json",
    "/iopsstor/scratch/cscs/dbartaula/experiments_assets_1/train_ootdiffusion/eval_metrics_curvton_extra_only.json",
]

if __name__ == "__main__":
    raise SystemExit(_main())

