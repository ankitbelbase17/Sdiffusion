import argparse
import json
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance

try:
    from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH
except Exception:
    CURVTON_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
    TRIPLET_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
    STREET_TRYON_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"
from eval_common import build_eval_loaders, evaluate_all_splits
from infer_variant import _build_datapred_model, _build_meanflow_model
from utils import make_beta_schedule, sample_ddim_like


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


def _log_and_validate_components(model, approach: str) -> None:
    print("Model components:")
    print(f"- approach: {approach}")
    print(f"- backbone: loaded ({type(model).__name__})")


def _resolve_diffusion_steps(args: argparse.Namespace, ckpt: dict | None) -> int:
    if args.diffusion_steps > 0:
        return int(args.diffusion_steps)
    if ckpt is not None:
        diff_cfg = ckpt.get("diffusion", {})
        if isinstance(diff_cfg, dict) and int(diff_cfg.get("steps", 0)) > 0:
            return int(diff_cfg["steps"])
    raise ValueError("Could not resolve diffusion_steps. Set --diffusion_steps > 0 or use a checkpoint with diffusion.steps metadata.")


def _to_01(x: torch.Tensor) -> torch.Tensor:
    if x.min() < -0.01 or x.max() > 1.01:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _to_u8(x01: torch.Tensor) -> torch.Tensor:
    return (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)


def _build_merged_loader(loaders, batch_size: int, num_workers: int) -> DataLoader:
    datasets = []
    for v in loaders.curvton.values():
        datasets.append(v.dataset)
    for v in loaders.triplet.values():
        datasets.append(v.dataset)
    for v in loaders.street.values():
        datasets.append(v.dataset)
    merged = ConcatDataset(datasets)
    return DataLoader(
        merged,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=loaders.curvton[next(iter(loaders.curvton))].collate_fn if loaders.curvton else None,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def _predict_datapred(model, device, batch_size: int, h: int, w: int, diffusion_steps: int):
    betas = make_beta_schedule(diffusion_steps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar)
    sqrt_1mab = torch.sqrt(1 - alpha_bar)
    sample_wide = sample_ddim_like(
        model=model,
        shape=(batch_size, 3, h, w),
        timesteps=diffusion_steps,
        sqrt_ab=sqrt_ab,
        sqrt_1mab=sqrt_1mab,
        device=device,
    )
    return sample_wide[:, :, :, :h]


@torch.no_grad()
def _predict_meanflow(model, device, batch_size: int, h: int, w: int, time_embed_scale: float):
    z1 = torch.randn(batch_size, 3, h, w, device=device)
    r = torch.zeros((batch_size,), device=device)
    t = torch.ones((batch_size,), device=device)
    u = model(z1, t * time_embed_scale, r * time_embed_scale)
    return (z1 - u)[:, :, :, :h]


def _build_predict_fn(args, model):
    @torch.no_grad()
    def _predict(batch, device):
        gt = _to_01(batch["ground_truth"].to(device))
        bs = gt.shape[0]
        h, w = gt.shape[-2], gt.shape[-1] * 2
        cloth = batch["cloth"].to(device)
        person = batch["person"].to(device)
        cond = torch.cat([person, cloth], dim=3)
        if args.approach == "datapred":
            betas = make_beta_schedule(args.diffusion_steps).to(device)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            sqrt_ab = torch.sqrt(alpha_bar)
            sqrt_1mab = torch.sqrt(1 - alpha_bar)
            pred = sample_ddim_like(
                model=model,
                shape=(bs, 3, h, w),
                timesteps=args._resolved_diffusion_steps,
                sqrt_ab=sqrt_ab,
                sqrt_1mab=sqrt_1mab,
                device=device,
                cond=cond,
            )[:, :, :, :h]
        else:
            pred = _predict_meanflow(model, device, bs, h, w, args.time_embed_scale)
        pred = _to_01(pred)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        return pred

    return _predict


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weight_source = "init_xavier"
    if args.approach == "datapred":
        model = _build_datapred_model(args).to(device).eval()
    else:
        model = _build_meanflow_model(args).to(device).eval()
    _log_and_validate_components(model, args.approach)

    ckpt = None
    if args.use_init_weights:
        print("Using initial custom DiT weights (no checkpoint load).")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=False)
        weight_source = f"checkpoint={args.checkpoint}"
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using init weights for evaluation.")
    print(f"Weights used: {weight_source}")
    args._resolved_diffusion_steps = _resolve_diffusion_steps(args, ckpt) if args.approach == "datapred" else args.diffusion_steps

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

    # Print full per-dataset metrics across CurvTON/Triplet/StreetTryOn splits.
    full_results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=_build_predict_fn(args, model),
        device=device,
        max_batches=args.max_batches,
        eval_frac_curvton=args.eval_frac_curvton,
        eval_frac_triplet=args.eval_frac_triplet,
        eval_frac_street=args.eval_frac_street,
    )

    merged = _build_merged_loader(loaders, args.batch_size, args.num_workers)

    est_n = len(merged.dataset)
    kid_subset = max(2, min(50, int(est_n)))
    fid = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=64, reset_real_features=True, normalize=True, subset_size=kid_subset).to(device)

    total_batches = len(merged)
    max_eval_batches = total_batches if args.max_batches <= 0 else min(total_batches, args.max_batches)
    n_images = 0

    for bidx, batch in enumerate(merged):
        if bidx >= max_eval_batches:
            break
        if batch is None:
            continue
        gt = _to_01(batch["ground_truth"].to(device))
        bs = gt.shape[0]
        if bs == 0:
            continue
        n_images += bs
        h, w = gt.shape[-2], gt.shape[-1] * 2
        if args.approach == "datapred":
            cloth = batch["cloth"].to(device)
            person = batch["person"].to(device)
            cond = torch.cat([person, cloth], dim=3)
            betas = make_beta_schedule(args._resolved_diffusion_steps).to(device)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            sqrt_ab = torch.sqrt(alpha_bar)
            sqrt_1mab = torch.sqrt(1 - alpha_bar)
            pred = sample_ddim_like(
                model=model,
                shape=(bs, 3, h, w),
                timesteps=args._resolved_diffusion_steps,
                sqrt_ab=sqrt_ab,
                sqrt_1mab=sqrt_1mab,
                device=device,
                cond=cond,
            )[:, :, :, :h]
        else:
            pred = _predict_meanflow(model, device, bs, h, w, args.time_embed_scale)
        pred = _to_01(pred)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        fid.update(_to_u8(gt), real=True)
        fid.update(_to_u8(pred), real=False)
        kid.update(_to_u8(gt), real=True)
        kid.update(_to_u8(pred), real=False)

    fid_val = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    out: Dict[str, float] = {
        "n_images": int(n_images),
        "fid": float(kid_mean.new_tensor(fid_val).item()),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "all_dataset_metrics": full_results,
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate custom DiT variants with FID/KID only")
    p.add_argument("--approach", type=str, required=True, choices=["datapred", "meanflow"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--image_width", type=int, default=-1)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=1280)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--num_heads", type=int, default=20)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--diffusion_steps", type=int, default=-1,
                   help=">0 overrides schedule steps. <=0 uses checkpoint diffusion.steps for datapred.")
    p.add_argument("--time_embed_scale", type=float, default=1000.0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--eval_frac_curvton", type=float, default=0.02)
    p.add_argument("--eval_frac_triplet", type=float, default=0.02)
    p.add_argument("--eval_frac_street", type=float, default=0.02)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_init_weights", action="store_true", default=False)
    p.add_argument("--output_json", type=str, default=None)
    main(p.parse_args())





