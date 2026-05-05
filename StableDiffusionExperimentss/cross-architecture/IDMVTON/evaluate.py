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
from train_idm_vton_local import IDMVTONModel  # noqa: E402


def _clean_state_dict(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        out[nk] = v
    return out


def _log_and_validate_components(model) -> None:
    required = ["vae", "scheduler", "unet", "image_encoder", "image_proj_model"]
    print("Model components:")
    for name in required:
        comp = getattr(model, name, None)
        if comp is None:
            raise RuntimeError(f"Required component missing or failed to load: {name}")
        print(f"- {name}: loaded ({type(comp).__name__})")


def build_predict_fn(model, num_inference_steps: int):
    @torch.no_grad()
    def _predict(batch, device):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        person_lat = model.encode(person)
        pose_lat = model.encode(person)
        cloth_lat = model.encode(cloth)
        person_mask = torch.zeros(
            person.shape[0], 1, person.shape[2], person.shape[3],
            device=person.device, dtype=person.dtype
        )
        person_mask = F.interpolate(person_mask, size=person_lat.shape[-2:], mode="bilinear", align_corners=False)
        latents = torch.randn_like(person_lat)
        model.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in model.scheduler.timesteps:
            t_batch = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
            noise_pred = model(latents, person_mask, person_lat, pose_lat, cloth, cloth_lat, t_batch)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        return model.vae.decode(latents / model.vae.config.scaling_factor).sample

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
    model = IDMVTONModel(args).to(device).eval()
    _log_and_validate_components(model)

    weight_source = "init_xavier"
    if args.use_init_weights:
        print("Using initial IDM-VTON weights (no checkpoint load).")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.unet.load_state_dict(_clean_state_dict(ckpt["unet_state_dict"]), strict=False)
        if "image_proj_state_dict" in ckpt:
            model.image_proj_model.load_state_dict(_clean_state_dict(ckpt["image_proj_state_dict"]), strict=False)
        weight_source = f"checkpoint={args.checkpoint}"
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using initial IDM-VTON weights (no checkpoint load).")
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
    feature_cache_root = _resolve_feature_cache_root(args)
    print(f"- Feature cache dir: {feature_cache_root}")
    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(model, args.num_inference_steps),
        device=device,
        max_batches=args.max_batches,
        eval_frac_curvton=args.eval_frac_curvton,
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
    p = argparse.ArgumentParser("Evaluate IDM-VTON on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    p.add_argument("--pretrained_garmentnet_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--image_encoder_path", type=str, default="ckpt/image_encoder")
    p.add_argument("--num_tokens", type=int, default=16)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_idm_vton")
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
    p.add_argument("--eval_frac_triplet", type=float, default=0.02)
    p.add_argument("--eval_frac_street", type=float, default=0.02)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cuda_device", type=int, default=None, help="CUDA device index (e.g., 1 -> cuda:1). Ignored if --device is set.")
    p.add_argument("--feature_cache_root", type=str, default="/iopsstor/scratch/cscs/dbartaula/featurecache")
    p.add_argument("--feature_cache_dir", type=str, default=None, help="Optional explicit feature-cache directory for this eval run")
    p.add_argument("--output_json", type=str, default=None)
    main(p.parse_args())







