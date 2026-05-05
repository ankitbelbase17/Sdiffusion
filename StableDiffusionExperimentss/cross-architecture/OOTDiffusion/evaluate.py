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
from train_ootdiffusion_local import OOTDiffusionModel  # noqa: E402

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
    required = ["vae", "scheduler", "denoising_unet", "outfitting_unet"]
    print("Model components:")
    for name in required:
        comp = getattr(model, name, None)
        if comp is None:
            raise RuntimeError(f"Required component missing or failed to load: {name}")
        print(f"- {name}: loaded ({type(comp).__name__})")


def build_predict_fn(model, num_inference_steps: int, is_masked: bool):
    @torch.no_grad()
    def _predict(batch, device):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        # When a mask is available (masked-variant training), apply the same
        # masking as training: person * (1 - mask) + 0 * mask (mid-grey in [-1, 1]).
        # Without a mask (triplet / street datasets), use person as-is
        # (equivalent to the unmasked training path where mask is all zeros).
        if is_masked and "mask" in batch:
            mask = batch["mask"].to(device)
            grey_fill = torch.zeros_like(person)
            person_for_model = torch.where(mask > 0.5, grey_fill, person)
        else:
            person_for_model = person
            mask = None
            mask_lat = None
        person_lat = model.encode(person_for_model)
        cloth_lat = model.encode(cloth)
        image_ori_lat = model.encode(person)
        if mask is not None:
            mask_lat = F.interpolate(mask, size=person_lat.shape[-2:]).to(
                device=device, dtype=person_lat.dtype
            )
        latents = torch.randn_like(person_lat)
        noise = latents.clone()
        model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = model.scheduler.timesteps
        for i, t in enumerate(timesteps):
            t_batch = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
            noise_pred = model(latents, person_lat, cloth_lat, t_batch)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
            if mask_lat is not None:
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents = model.scheduler.add_noise(
                        image_ori_lat,
                        noise,
                        torch.tensor([noise_timestep], device=device),
                    )
                else:
                    init_latents = image_ori_lat
                latents = (1.0 - mask_lat) * init_latents + mask_lat * latents
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
    model = OOTDiffusionModel(args.model_name, outfitting_dropout=0.0).to(device).eval()
    _log_and_validate_components(model)

    weight_source = "init_xavier"
    if args.use_init_weights:
        print("Using initial OOTDiffusion weights (no checkpoint load).")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.denoising_unet.load_state_dict(_clean_state_dict(ckpt["denoising_unet_state_dict"]), strict=False)
        model.outfitting_unet.load_state_dict(_clean_state_dict(ckpt["outfitting_unet_state_dict"]), strict=False)
        weight_source = f"checkpoint={args.checkpoint}"
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using initial OOTDiffusion weights (no checkpoint load).")
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
    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(model, args.num_inference_steps, args.is_masked),
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
    p = argparse.ArgumentParser("Evaluate OOTDiffusion on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_ootdiffusion")
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
    p.add_argument("--is_masked", action="store_true", help="Enable masked evaluation (uses repaint logic)")
    main(p.parse_args())







