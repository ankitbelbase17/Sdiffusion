import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(__file__)
STABLE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if STABLE_DIR not in sys.path:
    sys.path.insert(0, STABLE_DIR)

try:
    from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH  # noqa: E402
except Exception:
    CURVTON_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
    TRIPLET_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
    STREET_TRYON_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"
from eval_common import build_eval_loaders, evaluate_all_splits  # noqa: E402
from model import DiT250M, DiTConfig  # noqa: E402
from utils import make_beta_schedule, sample_ddim_like  # noqa: E402


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
    required = ["cfg"]
    print("Model components:")
    for name in required:
        comp = getattr(model, name, None)
        if comp is None:
            raise RuntimeError(f"Required component missing or failed to load: {name}")
        print(f"- {name}: loaded ({type(comp).__name__})")
    print(f"- backbone: loaded ({type(model).__name__})")


def _load_cfg(ckpt_cfg: dict) -> DiTConfig:
    return DiTConfig(
        image_size=ckpt_cfg.get("image_size", 64),
        image_height=ckpt_cfg.get("image_height", ckpt_cfg.get("image_size", 64)),
        image_width=ckpt_cfg.get("image_width", ckpt_cfg.get("image_size", 64) * 2),
        in_channels=ckpt_cfg.get("in_channels", 3),
        cond_in_channels=ckpt_cfg.get("cond_in_channels", 3),
        out_channels=ckpt_cfg.get("out_channels", 3),
        patch_size=ckpt_cfg.get("patch_size", 2),
        hidden_size=ckpt_cfg.get("hidden_size", 1280),
        depth=ckpt_cfg.get("depth", 9),
        num_heads=ckpt_cfg.get("num_heads", 20),
        mlp_ratio=ckpt_cfg.get("mlp_ratio", 4.0),
    )


def _resolve_diffusion_steps(args: argparse.Namespace, ckpt: dict | None) -> int:
    if args.diffusion_steps > 0:
        return int(args.diffusion_steps)
    if ckpt is not None:
        diff_cfg = ckpt.get("diffusion", {})
        if isinstance(diff_cfg, dict) and int(diff_cfg.get("steps", 0)) > 0:
            return int(diff_cfg["steps"])
    raise ValueError("Could not resolve diffusion_steps. Set --diffusion_steps > 0 or use a checkpoint with diffusion.steps metadata.")


def build_predict_fn(model, diffusion_steps: int, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor):
    @torch.no_grad()
    def _predict(batch, device):
        gt = batch["ground_truth"].to(device)
        cloth = batch["cloth"].to(device)
        person = batch["person"].to(device)
        cond = torch.cat([person, cloth], dim=3)
        bsz = gt.shape[0]
        sample = sample_ddim_like(
            model=model,
            shape=(bsz, model.cfg.out_channels if model.cfg.out_channels is not None else 3, cond.shape[-2], cond.shape[-1]),
            timesteps=diffusion_steps,
            sqrt_ab=sqrt_ab,
            sqrt_1mab=sqrt_1mab,
            device=device,
            cond=cond,
        )
        sample = sample[:, :, :, :gt.shape[-1]]
        if sample.shape[-2:] != gt.shape[-2:]:
            sample = F.interpolate(sample, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        return sample

    return _predict


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weight_source = "init_xavier"
    if args.use_init_weights:
        cfg = DiTConfig()
        ckpt = None
        print("Using initial custom DiT weights (no checkpoint load).")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = _load_cfg(ckpt.get("cfg", {}))
        weight_source = f"checkpoint={args.checkpoint}"
    else:
        cfg = DiTConfig()
        ckpt = None
        print("Using initial custom DiT weights (no checkpoint load).")
    model = DiT250M(cfg).to(device).eval()
    _log_and_validate_components(model)
    if ckpt is not None:
        model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=True)
        print(f"Loaded custom DiT checkpoint: {args.checkpoint}")
    else:
        print("Using initial custom DiT weights (no checkpoint load).")
    print(f"Weights used: {weight_source}")

    diffusion_steps = _resolve_diffusion_steps(args, ckpt)
    betas = make_beta_schedule(diffusion_steps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar)
    sqrt_1mab = torch.sqrt(1 - alpha_bar)
    if sqrt_ab is None or sqrt_1mab is None:
        raise RuntimeError("Required diffusion schedule tensors failed to initialize.")
    print("- diffusion_schedule: loaded (sqrt_ab, sqrt_1mab)")

    loaders = build_eval_loaders(
        curvton_test_data_path=args.curvton_test_data_path,
        triplet_test_data_path=args.triplet_test_data_path,
        street_tryon_data_path=args.street_tryon_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gender=args.gender,
        street_split=args.street_split,
    )
    print(f"- diffusion_steps: {diffusion_steps}")
    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(model, diffusion_steps, sqrt_ab, sqrt_1mab),
        device=device,
        max_batches=args.max_batches,
        eval_frac_curvton=args.eval_frac_curvton,
        eval_frac_triplet=args.eval_frac_triplet,
        eval_frac_street=args.eval_frac_street,
    )
    print("\nEvaluation metrics:\n" + json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate custom DiT pretraining model")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--diffusion_steps", type=int, default=-1,
                   help=">0 overrides schedule steps. <=0 uses checkpoint diffusion.steps.")
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--max_batches", type=int, default=0, help="0 = full dataset")
    p.add_argument("--eval_frac_curvton", type=float, default=0.02)
    p.add_argument("--eval_frac_triplet", type=float, default=0.02)
    p.add_argument("--eval_frac_street", type=float, default=0.02)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_init_weights", action="store_true", default=False)
    p.add_argument("--output_json", type=str, default=None)
    main(p.parse_args())




