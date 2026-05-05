import argparse
import json
from pathlib import Path
from datetime import datetime
import os

import torch

try:
    from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH
except Exception:
    CURVTON_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
    TRIPLET_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
    STREET_TRYON_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"
from eval_common import build_eval_loaders, evaluate_all_splits
from model import SDModel
from utils import decode_latents, run_full_inference


def _log_and_validate_components(model) -> None:
    # CATVTON path in this repo uses image-latent conditioning and does not
    # require text encoder/tokenizer at runtime.
    required = ["vae", "unet", "scheduler"]
    print("Model components:")
    for name in required:
        comp = getattr(model, name, None)
        if comp is None:
            raise RuntimeError(f"Required component missing or failed to load: {name}")
        print(f"- {name}: loaded ({type(comp).__name__})")


def _load_unet_checkpoint(unet, checkpoint_path: str, device: torch.device):
    # Load checkpoint on CPU first to avoid GPU OOM spikes during deserialization.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "unet_state_dict" in ckpt:
        sd = ckpt["unet_state_dict"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    clean = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        clean[nk] = v
    unet.load_state_dict(clean, strict=False)
    return ckpt.get("step", None)


def build_predict_fn(model, num_inference_steps: int, ootd: bool, decode_batch_size: int, vae_fp16_decode: bool):
    @torch.no_grad()
    def _predict(batch, device):
        cloth = batch["cloth"].to(device)
        person_img = batch["person"].to(device)
        cond_input = cloth if ootd else torch.cat([person_img, cloth], dim=3)
        cond_latents = model.vae.encode(cond_input).latent_dist.sample() * 0.18215
        pred_latents = run_full_inference(model, cond_latents, num_inference_steps=num_inference_steps)
        pred_wide = decode_latents(
            model.vae,
            pred_latents,
            decode_batch_size=decode_batch_size,
            vae_fp16=vae_fp16_decode,
        )
        if ootd:
            return pred_wide * 2 - 1
        return pred_wide[:, :, :, : cloth.shape[-1]] * 2 - 1

    return _predict


def _resolve_feature_cache_root(args):
    root = args.feature_cache_root
    if args.feature_cache_dir:
        return args.feature_cache_dir
    if args.use_init_weights:
        run_name = getattr(args, "run_name", None) or "init_weights"
    elif args.checkpoint:
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
    model = SDModel().to(device)
    _log_and_validate_components(model)
    model.unet.eval()
    weight_source = "init_xavier"
    if args.use_init_weights:
        print("Using initial CATVTON weights (no checkpoint load).")
    elif args.checkpoint:
        step = _load_unet_checkpoint(model.unet, args.checkpoint, device)
        weight_source = f"checkpoint={args.checkpoint}"
        if step is None:
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"Loaded checkpoint: {args.checkpoint} (step={step})")
    else:
        print("Using initial CATVTON weights (no checkpoint load).")
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
        size=args.image_size,
        gender=args.gender,
        street_split=args.street_split,
    )
    print(f"- CurvTON splits discovered: {sorted(loaders.curvton.keys())}")
    if args.curvton_splits:
        allowed = {f"curvton_{s.strip().lower()}" for s in args.curvton_splits.split(",") if s.strip()}
        missing = sorted(k for k in allowed if k not in loaders.curvton)
        loaders.curvton = {k: v for k, v in loaders.curvton.items() if k in allowed}
        print(f"- CurvTON splits filter: {sorted(allowed)}")
        if missing:
            print(f"- WARNING: requested CurvTON splits not found in dataset: {missing}")
        print(f"- CurvTON splits to evaluate: {sorted(loaders.curvton.keys())}")
    feature_cache_root = _resolve_feature_cache_root(args)
    print(f"- Feature cache dir: {feature_cache_root}")
    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(
            model,
            args.num_inference_steps,
            args.ootd,
            args.decode_batch_size,
            args.vae_fp16_decode,
        ),
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
    p = argparse.ArgumentParser("Evaluate CATVTON on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=512, help="<=0 keeps native resolution (no resize)")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--decode_batch_size", type=int, default=1, help="Chunk size for VAE decode to reduce VRAM.")
    p.add_argument("--vae_fp16_decode", action="store_true", default=True, help="Use fp16 autocast during VAE decode.")
    p.add_argument("--no_vae_fp16_decode", action="store_false", dest="vae_fp16_decode",
                   help="Disable fp16 decode autocast.")
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--max_batches", type=int, default=0, help="0 = full dataset")
    p.add_argument("--eval_frac_curvton", type=float, default=0.02)
    p.add_argument(
        "--eval_frac_curvton_overall",
        type=float,
        default=None,
        help="Eval fraction for curvton_overall split. If not set, uses --eval_frac_curvton.",
    )
    p.add_argument(
        "--eval_frac_curvton_extra",
        type=float,
        default=0.02,
        help="Eval fraction for CurvTON extra semantic splits "
             "(traditional/non_traditional and dresses/upper_body/lower_body).",
    )
    p.add_argument("--eval_frac_triplet", type=float, default=0.02)
    p.add_argument("--eval_frac_street", type=float, default=0.02)
    p.add_argument(
        "--curvton_splits",
        type=str,
        default=None,
        help="Comma-separated CurvTON split names from: easy,medium,hard,overall",
    )
    p.add_argument("--ootd", action="store_true", default=False)
    p.add_argument("--use_init_weights", action="store_true", default=False)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cuda_device", type=int, default=None, help="CUDA device index (e.g., 1 -> cuda:1). Ignored if --device is set.")
    p.add_argument("--feature_cache_root", type=str, default="/iopsstor/scratch/cscs/dbartaula/featurecache")
    p.add_argument("--feature_cache_dir", type=str, default=None, help="Optional explicit feature-cache directory for this eval run")
    p.add_argument("--output_json", type=str, default=None)
    main(p.parse_args())






