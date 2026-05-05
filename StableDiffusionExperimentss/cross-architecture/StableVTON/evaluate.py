import argparse
import json
from pathlib import Path
from datetime import datetime
import os
import sys

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(__file__)
STABLE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in (STABLE_DIR, CROSS_ARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH  # noqa: E402
except Exception:
    CURVTON_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
    TRIPLET_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
    STREET_TRYON_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"
from eval_common import build_eval_loaders, evaluate_all_splits  # noqa: E402

SLURM_VITON_DIR = "/iopsstor/scratch/cscs/dbartaula/StableVITON"
LOCAL_VITON_DIR = os.path.abspath(os.path.join(STABLE_DIR, "..", "StableVITON"))
STABLE_VITON_DIR = SLURM_VITON_DIR if os.path.exists(SLURM_VITON_DIR) else LOCAL_VITON_DIR
if STABLE_VITON_DIR not in sys.path:
    sys.path.insert(0, STABLE_VITON_DIR)

from omegaconf import OmegaConf
from cldm.model import create_model

DEFAULT_CURVTON_SPLITS = [
    "easy",
    "medium",
    "hard",
    "overall",
    "traditional",
    "non_traditional",
    "dresses",
    "upper_body",
    "lower_body",
]

_CURVTON_SPLIT_TO_LOADER = {
    "easy": "curvton_easy",
    "medium": "curvton_medium",
    "hard": "curvton_hard",
    "overall": "curvton_overall",
    "traditional": "curvton_traditional",
    "non_traditional": "curvton_non_traditional",
    "dresses": "curvton_dresses",
    "upper_body": "curvton_upper_body",
    "lower_body": "curvton_lower_body",
}


def build_predict_fn(model, num_inference_steps: int, is_masked: bool):
    @torch.no_grad()
    def _predict(batch, device):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        
        # Ground truth is required for the official dict, but isn't used for inference
        gt = batch.get("ground_truth", person).to(device) 
        
        if is_masked and "mask" in batch:
            mask = batch["mask"].to(device)
            if "pose" in batch:
                pose = batch["pose"].to(device)
            else:
                pose = person # Surrogate pose
            grey_fill = torch.full_like(person, 0.5)
            agnostic = torch.where(mask > 0.5, grey_fill, person)
            agn_mask = mask
            image_densepose = pose
        else:
            # Unmasked logic
            agnostic = person
            agn_mask = torch.zeros(person.shape[0], 1, person.shape[2], person.shape[3], device=device, dtype=person.dtype)
            image_densepose = torch.zeros_like(person)
            if "mask" in batch:
                mask = batch["mask"].to(device)
            else:
                mask = agn_mask

        official_batch = {
            "image": gt,
            "cloth": cloth,
            "agn": agnostic,
            "agn_mask": agn_mask,
            "image_densepose": image_densepose,
            "txt": [""] * person.shape[0],
            "gt_cloth_warped_mask": mask,
        }

        # Predict using official log_images to ensure consistent behavior
        log_dict = model.log_images(
            official_batch,
            N=person.shape[0],
            sample=True,
            unconditional_guidance_scale=5.0,
            ddim_steps=num_inference_steps,
        )
        
        pred_key = f"samples_cfg_scale_5.00"
        if pred_key in log_dict:
            preds = log_dict[pred_key]
        else:
            preds = log_dict["samples"]
            
        return preds

    return _predict


def _resolve_feature_cache_root(args):
    root = args.feature_cache_root
    if args.feature_cache_dir:
        return args.feature_cache_dir
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        run_name = ckpt.parent.parent.name if ckpt.parent.name == "checkpoints" else ckpt.parent.name
    else:
        run_name = getattr(args, "run_name", None) or "init_weights"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(root) / run_name / f"eval_{stamp}")


def _resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if args.cuda_device is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.cuda_device}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    device = _resolve_device(args)
    
    config_path = os.path.join(STABLE_VITON_DIR, "configs", "VITONHD.yaml")
    config = OmegaConf.load(config_path)
    model = create_model(config_path, config=config).to(device)
    model.eval()

    weight_source = "init_weights"
    if args.use_init_weights:
        print("Using initial StableVITON weights (no checkpoint load).")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        weight_source = f"checkpoint={args.checkpoint}"
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using initial StableVITON weights (no checkpoint load).")
        
    print(f"Weights used: {weight_source}")
    print("\nDatasets for evaluation:")
    print(f"- CurvTON test: {args.curvton_test_data_path}")
    print(f"- Triplet test: {args.triplet_test_data_path}")
    print(f"- StreetTryOn ({args.street_split}): {args.street_tryon_data_path}")
    loaders = build_eval_loaders(
        curvton_test_data_path=args.curvton_test_data_path,
        triplet_test_data_path=args.triplet_test_data_path,
        street_tryon_data_path=args.street_tryon_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gender=args.gender,
        street_split=args.street_split,
    )
    if args.curvton_splits:
        requested = [s.strip().lower() for s in args.curvton_splits.split(",") if s.strip()]
        keep_keys = []
        for split in requested:
            key = _CURVTON_SPLIT_TO_LOADER.get(split)
            if key is not None:
                keep_keys.append(key)
        loaders.curvton = {k: v for k, v in loaders.curvton.items() if k in keep_keys}
    feature_cache_root = _resolve_feature_cache_root(args)
    print(f"- Feature cache dir: {feature_cache_root}")
    
    # Check if masked eval is requested
    is_masked = args.is_masked or ("mask" in args.run_name.lower())
    
    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(model, args.num_inference_steps, is_masked),
        device=device,
        max_batches=args.max_batches,
        eval_frac_curvton=args.eval_frac_curvton,
        eval_frac_curvton_overall=args.eval_frac_curvton_overall,
        eval_frac_curvton_extra=args.eval_frac_curvton_extra,
        eval_frac_triplet=args.eval_frac_triplet,
        eval_frac_street=args.eval_frac_street,
        feature_cache_root=feature_cache_root,
    )
    print("\nEvaluation metrics:\n" + json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate StableVITON on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_stable_vton")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--use_init_weights", action="store_true", default=False)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full dataset")
    p.add_argument("--eval_frac_curvton", type=float, default=0.02)
    p.add_argument("--eval_frac_curvton_overall", type=float, default=None)
    p.add_argument("--eval_frac_curvton_extra", type=float, default=0.02)
    p.add_argument("--eval_frac_triplet", type=float, default=0.02)
    p.add_argument("--eval_frac_street", type=float, default=0.02)
    p.add_argument("--curvton_splits", type=str, default=",".join(DEFAULT_CURVTON_SPLITS))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cuda_device", type=int, default=None, help="CUDA device index (e.g., 1 -> cuda:1). Ignored if --device is set.")
    p.add_argument("--feature_cache_root", type=str, default="/iopsstor/scratch/cscs/dbartaula/featurecache")
    p.add_argument("--feature_cache_dir", type=str, default=None, help="Optional explicit feature-cache directory for this eval run")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--is_masked", action="store_true", help="Set to use the masked pipeline")
    main(p.parse_args())
