"""
train_flux.py — HunyuanDiT v1.1 training for CatVTON-style virtual try-on.

Conditioning approach (CatVTON-style spatial concatenation)
───────────────────────────────────────────────────────────
  cond_lat    = VAE( cat([person, cloth], W) )    [B, 4, H, 2W]
  target_lat  = VAE( cat([gt,     cloth], W) )    [B, 4, H, 2W]
  noisy       = DDPM.add_noise(target_lat, ε, t)  [B, 4, H, 2W]
  full_input  = cat([noisy, cond_lat], C)          [B, 8, H, 2W]
  → HunyuanDiT2DModel (2D input, NO packing) → take first 4 channels → ε loss

DDPM epsilon-prediction training:
  noisy  = scheduler.add_noise(clean, ε, t)      t ~ U[0, 1000]
  target = ε   (model predicts the noise)
  loss   = MSE(predicted_ε, true_ε)

RoPE positional embeddings are computed for latent spatial size H×2W.
Supports hard / soft / reverse curriculum training on CurvTon.

PREREQUISITES:
  • Public model — no gated license required.
  • ~10–14 GB VRAM in float16 with gradient checkpointing.
"""

import os
import glob
import math
import collections
import statistics
import random
import argparse
import traceback
import logging
import datetime

_train_log = logging.getLogger("train_dit")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import weave

try:
    from config import (
        WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY,
        HUNYUAN_MODEL_NAME, IMAGE_SIZE,
    )
except Exception:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Stable_diffusion")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
    HUNYUAN_MODEL_NAME = os.getenv("HUNYUAN_MODEL_NAME", "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))
from hunyuan_model import (
    HunyuanDiTModel,
    freeze_non_attention_hunyuan,
    print_trainable_params_hunyuan,
)
from utils import (
    CurvtonDataset,
    CombinedCurvtonDataset,
    collate_fn,
    get_curvton_test_dataloaders,
    get_triplet_test_dataloaders,
    get_triplet_train_loader,
    _CURRIC_STAGES,
    curriculum_weights  as _curriculum_weights,
    subsample_dataset   as _subsample_dataset,
    _POSE_AVAILABLE,
    _pose_keypoint_error,
)

class _AttentionActivationRegularizer:
    """Collect attention output magnitude as a proxy for unnecessary attention."""

    def __init__(self, module):
        self._handles = []
        self._vals = []
        for name, subm in module.named_modules():
            if not hasattr(subm, "to_q"):
                continue
            if (".attn1" not in name) and (".attn2" not in name):
                continue
            self._handles.append(subm.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        if torch.is_tensor(out):
            self._vals.append(out.float().pow(2).mean())
        elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            self._vals.append(out[0].float().pow(2).mean())

    def clear(self):
        self._vals.clear()

    def penalty(self, device):
        if not self._vals:
            return torch.zeros((), device=device)
        return torch.stack(self._vals).mean()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

try:
    import lpips as lpips_lib
except ImportError:
    raise ImportError("pip install lpips")
try:
    from torchmetrics.image import (
        FrechetInceptionDistance,
        KernelInceptionDistance,
        StructuralSimilarityIndexMeasure,
        PeakSignalNoiseRatio,
    )
except ImportError:
    raise ImportError("pip install torchmetrics[image]")


# ============================================================
# HUNYUANDIT INFERENCE / LOGGING / EVALUATION
# ============================================================

@torch.no_grad()
def run_full_inference_hunyuan(
    model: HunyuanDiTModel,
    cond_latents: torch.Tensor,
    num_inference_steps: int = 50,
) -> torch.Tensor:
    """
    Full denoising loop with UniPC scheduler (HunyuanDiT v1.1).

    Args:
        model:               HunyuanDiTModel (vae + transformer + schedulers)
        cond_latents:        conditioning latent [B, 4, H, W_cat]
        num_inference_steps: number of UniPC steps (default 50)

    Returns:
        denoised latents [B, 4, H, W_cat]  (same spatial shape as cond_latents)
    """
    device = cond_latents.device
    dtype  = cond_latents.dtype
    B, C, H_lat, W_lat = cond_latents.shape

    # Start from pure noise (target shape)
    latents = torch.randn(B, C, H_lat, W_lat, device=device, dtype=dtype)

    # Channel concat: noisy + cond -> [B, 8, H, W]
    W_full = W_lat

    # Zero text conditioning (T5 + CLIP, 1 token each)
    txt_emb  = torch.zeros(B, 1, model.cross_attn_dim, device=device, dtype=dtype)
    txt_mask = torch.ones(B, 1, device=device, dtype=torch.bool)

    # image_meta_size [B, 6]: (orig_H, orig_W, crop_top, crop_left, tgt_H, tgt_W) in pixels
    px_scale = 8 if model.use_vae else 1
    meta_size = torch.tensor(
        [[H_lat * px_scale, W_full * px_scale, 0, 0, H_lat * px_scale, W_full * px_scale]] * B,
        device=device, dtype=dtype,
    )
    style_ids = torch.zeros(B, device=device, dtype=torch.long)

    # RoPE embeddings for the full-width latent
    rope_cos, rope_sin = model.get_rope_embed(H_lat, W_full, device, dtype)

    model.inference_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in model.inference_scheduler.timesteps:
        full_lat = torch.cat([latents, cond_latents], dim=1)   # [B, 8, H, W]

        noise_pred_full = model.transformer(
            hidden_states=full_lat,
            timestep=t.expand(B),
            encoder_hidden_states=txt_emb,
            text_embedding_mask=txt_mask,
            encoder_hidden_states_t5=txt_emb,
            text_embedding_mask_t5=txt_mask,
            image_meta_size=meta_size,
            style=style_ids,
            image_rotary_emb=(rope_cos, rope_sin),
            return_dict=False,
        )[0]

        noise_pred = noise_pred_full[:, :C]
        latents = model.inference_scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def log_images_hunyuan(step, batch, model: HunyuanDiTModel,
                       noisy_latents, noise_pred, cond_latents, target_latents,
                       num_inference_steps=50):
    """Log sample images to W&B during HunyuanDiT training."""

    def _to_wandb(tensor, caption):
        img = (tensor[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        return wandb.Image(img, caption=caption)

    W = IMAGE_SIZE  # pixel width of one image (512)

    with torch.no_grad():
        print(f"\n🔄 HunyuanDiT inference ({num_inference_steps} steps) ...")
        model.transformer.eval()
        full_inf_lat = run_full_inference_hunyuan(
            model, cond_latents, num_inference_steps,
        )
        model.transformer.train()
        print("✓ HunyuanDiT inference complete")

        def _tryon(lat):
            decoded = model.decode_latent(lat)          # [B,3,H,2W]
            return decoded[:, :, :, :W]                 # left half = try-on

        full_inf_img   = _tryon(full_inf_lat)
        target_img     = _tryon(target_latents)
        noisy_img      = _tryon(noisy_latents)

        # single-step noise residual approximation
        denoised_est     = noisy_latents - noise_pred
        denoised_est_img = _tryon(denoised_est)

        cond_img = model.decode_latent(cond_latents)    # [B,3,H,2W]

        gt    = (batch["ground_truth"][0:1] + 1) / 2
        cloth = (batch["cloth"][0:1] + 1) / 2
        _pk   = "person" if "person" in batch else "masked_person"
        person_vis = (batch[_pk][0:1] + 1) / 2

    wandb.log({
        "images/ground_truth":        _to_wandb(gt,              "Ground Truth"),
        "images/cloth":               _to_wandb(cloth,           "Cloth"),
        "images/person":              _to_wandb(person_vis,      "Person"),
        "images/cond_decoded":        _to_wandb(cond_img,        "Cond decoded"),
        "images/target_decoded":      _to_wandb(target_img,      "Target decoded"),
        "images/noisy_decoded":       _to_wandb(noisy_img,       "Noisy decoded"),
        "images/denoised_single_step":_to_wandb(denoised_est_img,"Single-step denoised"),
        "images/full_inference":      _to_wandb(full_inf_img,    f"Full Inference ({num_inference_steps} steps)"),
    }, step=step)


@torch.no_grad()
def evaluate_on_test_hunyuan(model: HunyuanDiTModel, test_loaders, device,
                             num_inference_steps=50, eval_frac=0.10,
                             ootd=False):
    """
    Compute LPIPS, SSIM, PSNR, FID, KID, (PKE) on each test split.
    """
    model.transformer.eval()
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    log_dict: dict[str, float] = {}

    for split_name, loader in test_loaders.items():
        n_batches = max(1, math.ceil(len(loader) * eval_frac))
        print(f"\n[Eval-HunyuanDiT] {split_name}: {n_batches}/{len(loader)} batches …")

        lpips_vals, ssim_vals, psnr_vals = [], [], []
        pke_vals: list[float] = []

        fid = FrechetInceptionDistance(feature=2048, reset_real_features=True,
                                       normalize=True).to(device)
        kid = KernelInceptionDistance(feature=2048, reset_real_features=True,
                                      normalize=True, subset_size=50).to(device)
        ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            gt    = batch["ground_truth"].to(device)
            cloth = batch["cloth"].to(device)
            person_img = batch["person"].to(device)

            # Encode conditioning
            if ootd:
                cond_input = cloth
            else:
                cond_input = torch.cat([person_img, cloth], dim=3)
            cond_latents = model.encode_image(cond_input)

            # Full inference
            pred_latents = run_full_inference_hunyuan(
                model, cond_latents, num_inference_steps,
            )
            pred_wide  = model.decode_latent(pred_latents)
            pred_tryon = pred_wide[:, :, :, :IMAGE_SIZE]
            real_tryon = (gt / 2 + 0.5).clamp(0, 1)

            # Metrics
            lp = lpips_fn(pred_tryon * 2 - 1, real_tryon * 2 - 1)
            lpips_vals.extend(lp.view(-1).cpu().tolist())
            ssim_vals.append(ssim_m(pred_tryon, real_tryon).item())
            psnr_vals.append(psnr_m(pred_tryon, real_tryon).item())

            real_u8 = (real_tryon * 255).to(torch.uint8)
            pred_u8 = (pred_tryon * 255).to(torch.uint8)
            fid.update(real_u8, real=True);  fid.update(pred_u8, real=False)
            kid.update(real_u8, real=True);  kid.update(pred_u8, real=False)

            if _POSE_AVAILABLE:
                for b in range(real_tryon.shape[0]):
                    r_np = (real_tryon[b].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
                    p_np = (pred_tryon[b].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
                    pke = _pose_keypoint_error(r_np, p_np)
                    if pke is not None:
                        pke_vals.append(pke)

        fid_score = fid.compute().item()
        kid_mean, kid_std = kid.compute()

        lpips_arr = np.array(lpips_vals)
        ssim_arr  = np.array(ssim_vals)
        psnr_arr  = np.array(psnr_vals)

        log_dict[f"test/{split_name}/lpips_mean"] = float(lpips_arr.mean())
        log_dict[f"test/{split_name}/ssim_mean"]  = float(ssim_arr.mean())
        log_dict[f"test/{split_name}/psnr_mean"]  = float(psnr_arr.mean())
        log_dict[f"test/{split_name}/fid"]        = fid_score
        log_dict[f"test/{split_name}/kid_mean"]   = kid_mean.item()
        log_dict[f"test/{split_name}/kid_std"]    = kid_std.item()
        if pke_vals:
            log_dict[f"test/{split_name}/pke_mean"] = float(np.mean(pke_vals))

        print(f"   {split_name:6s} | LPIPS={lpips_arr.mean():.4f}  "
              f"SSIM={ssim_arr.mean():.4f}  PSNR={psnr_arr.mean():.2f}  "
              f"FID={fid_score:.2f}  KID={kid_mean.item():.4f}")

        del fid, kid, ssim_m, psnr_m
        torch.cuda.empty_cache()

    del lpips_fn; torch.cuda.empty_cache()
    model.transformer.train()
    return log_dict


# ============================================================
# MAIN TRAINING
# ============================================================

def train(args):
    # ── Distributed Data Parallel setup ────────────────────────
    rank       = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    is_main    = (rank == 0)

    if world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16       # HunyuanDiT was trained in fp16

    if is_main:
        wandb.login(key=WANDB_API_KEY)
        weave.init(f"{os.getenv('WANDB_ENTITY', WANDB_ENTITY)}/{os.getenv('WANDB_PROJECT', WANDB_PROJECT)}")
        print(f"Device: {device}  |  rank={rank}/{world_size}  |  dtype: {dtype}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if is_main:
            print("✓ cudnn.benchmark=True, TF32 enabled")

    # ── Dataset (CurvTon only for Flux try-on) ──────────────────
    if is_main:
        print(f"Data source: CurvTon from {args.curvton_data_path} "
              f"(difficulty={args.difficulty}, gender={args.gender})")
    genders = ("female", "male") if args.gender == "all" else (args.gender,)

    # ── Model ───────────────────────────────────────────────────
    model = HunyuanDiTModel(dtype=dtype, gradient_checkpointing=True, use_vae=True).to(device)

    if args.ootd:
        print("\n🔶 OOTDiffusion mode: cloth-only cond (no person/mask)")

    if args.train_mode == "attention_only":
        model.transformer = freeze_non_attention_hunyuan(model.transformer)
        _mode_tag = "attn"
    else:
        _mode_tag = "full"

    attn_score_reg = None
    if args.use_attention_score_regularization:
        attn_score_reg = _AttentionActivationRegularizer(model.transformer)

    _frac_tag   = f"_frac{int(args.data_fraction * 100)}pct" if args.data_fraction < 1.0 else ""
    _phase2_tag = "_phase2" if getattr(args, 'phase2_data_path', None) else ""
    _ootd_tag   = "_ootd"   if args.ootd else ""
    _auto_name  = (
        f"hunyuan_{_mode_tag}"
        f"_bs{args.batch_size}"
        f"_{args.curriculum}"
        f"_g{args.gender}"
        f"{_frac_tag}"
        f"_s{args.max_steps}"
        f"{_phase2_tag}"
        f"{_ootd_tag}"
    )
    run_name = args.run_name if args.run_name else _auto_name
    if is_main:
        print(f"\n\U0001f3f7  WandB run name: {run_name}")

    trainable_params = print_trainable_params_hunyuan(model, args.train_mode)

    if hasattr(torch, "compile"):
        if is_main:
            print("\n⚙  Compiling HunyuanDiT with torch.compile (mode='reduce-overhead') ...")
        model.transformer = torch.compile(model.transformer, mode="reduce-overhead")
        if is_main:
            print("   ✓ torch.compile applied to HunyuanDiT.")

    # ── DDP Wrap ────────────────────────────────────────────────
    if world_size > 1:
        model.transformer = DDP(
            model.transformer,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        if is_main:
            print(f"✓ DDP: {world_size} GPUs, total batch={args.batch_size * world_size}")

    # ── DataLoaders ─────────────────────────────────────────────
    diff_loaders:  dict = {}
    diff_samplers: dict = {}
    for _diff in CurvtonDataset.DIFFICULTIES:
        _ds = CombinedCurvtonDataset(
            root_dir=args.curvton_data_path,
            difficulties=(_diff,), genders=genders, size=IMAGE_SIZE,
        )
        _ds = _subsample_dataset(_ds, args.data_fraction)
        _sampler = DistributedSampler(_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        diff_samplers[_diff] = _sampler
        diff_loaders[_diff] = DataLoader(
            _ds, batch_size=args.batch_size,
            shuffle=(_sampler is None),
            sampler=_sampler,
            num_workers=args.num_workers, drop_last=True,
            collate_fn=collate_fn, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
        )
        if is_main:
            print(f"✓ CurvTon [{_diff}] DataLoader: {len(diff_loaders[_diff])} batches")

    difficulties = CurvtonDataset.DIFFICULTIES if args.difficulty == "all" else (args.difficulty,)
    train_dataset = CombinedCurvtonDataset(
        root_dir=args.curvton_data_path,
        difficulties=difficulties, genders=genders, size=IMAGE_SIZE,
    )
    train_dataset = _subsample_dataset(train_dataset, args.data_fraction)
    _frac_tag = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
    dataset_label = f"CurvTon-{args.difficulty}-{args.gender}-{args.curriculum}{_frac_tag}"

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    batches_per_epoch = len(train_loader)
    # Override stage_steps if --stage_epochs is set
    if args.stage_epochs > 0:
        args.stage_steps = args.stage_epochs * batches_per_epoch
        if is_main:
            print(f"✓ stage_epochs={args.stage_epochs} → stage_steps={args.stage_steps} ({batches_per_epoch} batches/epoch)")
    if is_main:
        print(f"✓ DataLoader ({dataset_label}): {batches_per_epoch} batches/epoch")

    # ── Experiments directory ────────────────────────────────────
    _EXPERIMENTS_ROOT = "/iopsstor/scratch/cscs/dbartaula/experiments_assets"
    experiments_dir = os.path.join(_EXPERIMENTS_ROOT, run_name)
    ckpt_dir = os.path.join(experiments_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"✓ Checkpoint dir: {ckpt_dir}")

    # ── Phase-2 loader (vitonhd + dresscode) — hard curriculum only ──
    phase2_loader  = None
    phase2_sampler = None
    if getattr(args, 'phase2_data_path', None):
        if is_main:
            print(f"\n[Phase2] Building combined triplet loader from {args.phase2_data_path} ...")
        try:
            phase2_loader, phase2_sampler = get_triplet_train_loader(
                root_dir=args.phase2_data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                size=IMAGE_SIZE,
                world_size=world_size,
                rank=rank,
            )
            if is_main:
                print(f"[Phase2] ✓ loader ready: {len(phase2_loader)} batches/epoch")
        except Exception:
            if is_main:
                print(f"[Phase2] WARNING: could not build phase2 loader:\n{traceback.format_exc()}")

    # ── Test loaders ────────────────────────────────────────
    test_loaders: dict = {}
    if args.curvton_test_data_path:
        test_loaders.update({
            f"curvton_{k}": v
            for k, v in get_curvton_test_dataloaders(
                root_dir=args.curvton_test_data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                size=IMAGE_SIZE,
                genders=("female", "male") if args.gender == "all" else (args.gender,),
            ).items()
        })
        if is_main:
            print(f"✓ CurvTon test loaders: {[k for k in test_loaders]}")
    elif is_main:
        print("[Eval] No --curvton_test_data_path; CurvTon test eval skipped.")
    if args.triplet_test_data_path:
        triplet_test = get_triplet_test_dataloaders(
            root_dir=args.triplet_test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
        )
        test_loaders.update(triplet_test)
        if is_main:
            print(f"✓ Triplet test loaders: {list(triplet_test.keys())}")
    else:
        if is_main:
            print("[Eval] No --triplet_test_data_path; triplet eval skipped.")
    if not test_loaders:
        test_loaders = None

    # ── Dataloader sanity check (all loaders except phase2) ─────────
    if is_main:
        _train_log.info("[Sanity] Starting dataloader sanity check …")
        print("\n── Dataloader sanity check ─────────────────────────────────")
    _sanity_loaders: dict = {"train": train_loader}
    for _sn, _sdl in (diff_loaders or {}).items():
        _sanity_loaders[f"diff/{_sn}"] = _sdl
    for _sn, _sdl in (test_loaders or {}).items():
        _sanity_loaders[f"test/{_sn}"] = _sdl
    _sanity_errors: list = []
    for _sn, _sdl in _sanity_loaders.items():
        try:
            _sb = next(iter(_sdl))
            if _sb is None:
                _sanity_errors.append(f"  [{_sn}] first batch returned None (all samples corrupt?)")
            else:
                for _req_key in ("person", "cloth", "ground_truth"):
                    if _req_key not in _sb:
                        _sanity_errors.append(f"  [{_sn}] missing required key '{_req_key}'")
                if is_main:
                    _shapes = {k: tuple(v.shape) for k, v in _sb.items() if hasattr(v, "shape")}
                    _train_log.info("[Sanity] %-30s  keys=%s  shapes=%s", _sn, list(_sb.keys()), _shapes)
                    print(f"  ✓ {_sn:30s}  shapes={_shapes}")
        except Exception as _se:
            _sanity_errors.append(
                f"  [{_sn}] raised {type(_se).__name__}: {_se}\n{traceback.format_exc()}"
            )
    if _sanity_errors:
        _msg = "\n".join(_sanity_errors)
        if is_main:
            _train_log.error("[Sanity] FAILED on rank %d:\n%s", rank, _msg)
            print(f"\n[FATAL] Dataloader sanity check FAILED:\n{_msg}", flush=True)
        raise RuntimeError(f"Dataloader sanity check failed (rank {rank}):\n" + _msg)
    if is_main:
        _train_log.info("[Sanity] All %d loaders OK.", len(_sanity_loaders))
        print(f"✓ Sanity check passed — {len(_sanity_loaders)} loaders OK.\n")
    if world_size > 1:
        dist.barrier()  # ensure all ranks pass before training starts

    # ── WandB ───────────────────────────────────────────────────
    if is_main:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", WANDB_PROJECT), entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
            id=run_name,
            resume="allow",
            config={
                "model":           HUNYUAN_MODEL_NAME,
                "architecture":    "HunyuanDiT v1.1 (CatVTON spatial concat)",
                "lr":              args.lr,
                "batch_size":      args.batch_size,
                "epochs":          args.epochs,
                "train_mode":      args.train_mode,
                "dataset":         dataset_label,
                "trainable_params":trainable_params,
                "curriculum":      args.curriculum,
                "stage_steps":     args.stage_steps,
                "data_fraction":   args.data_fraction,
                "ootd":            args.ootd,
                "num_inference_steps": args.num_inference_steps,
                "dtype":           str(dtype),
                "max_steps":       args.max_steps,
                "phase2_data_path":getattr(args, 'phase2_data_path', None),
                "phase2_start_step": getattr(args, 'phase2_start_step', 28801),
            },
            name=run_name,
        )

    # ── Optimizer & scheduler ───────────────────────────────────
    _transformer_raw = model.transformer.module if isinstance(model.transformer, DDP) else model.transformer
    trainable_params_list = [p for p in _transformer_raw.parameters() if p.requires_grad]
    optimizer    = AdamW(trainable_params_list, lr=args.lr)
    _total_steps = (
        args.max_steps if args.max_steps > 0
        else args.epochs * batches_per_epoch
    )
    _LR_ETA_MIN  = 1e-6
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(_total_steps, 1), eta_min=_LR_ETA_MIN)
    scaler       = GradScaler()

    global_step = 0
    start_epoch = 0
    ema_loss    = None

    # ── Checkpoint resume ───────────────────────────────────────
    ckpt_to_load = None
    if args.resume:
        ckpt_to_load = args.resume
        print(f"\n▶ --resume specified: {ckpt_to_load}")
    elif not args.no_resume:
        _exp_ckpt_dir = os.path.join("/iopsstor/scratch/cscs/dbartaula/experiments_assets", run_name, "checkpoints")
        candidates = (
            glob.glob(os.path.join(_exp_ckpt_dir, "ckpt_step_*.pt"))
            + glob.glob(os.path.join(_exp_ckpt_dir, "ckpt_final.pt"))
            + glob.glob(os.path.join(args.checkpoint_dir,
                                     f"checkpoint_hunyuan_{args.train_mode}_step_*.pt"))
        )
        if candidates:
            def _step_num(p):
                basename = os.path.basename(p)
                if basename == "ckpt_final.pt":
                    return float('inf')
                try:    return int(basename.split("_step_")[1].replace(".pt",""))
                except: return -1
            ckpt_to_load = max(candidates, key=_step_num)
            if is_main:
                print(f"\n▶ Auto-detected: {ckpt_to_load}")
    else:
        print("\n▶ --no_resume set: training from scratch.")

    if ckpt_to_load:
        print("   Loading checkpoint …")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        model.transformer.load_state_dict(ckpt["transformer_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(global_step):
                lr_scheduler.step()
        if "ema_loss" in ckpt and ckpt["ema_loss"] is not None:
            ema_loss = ckpt["ema_loss"]
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"   ✓ Resumed — step={global_step}, epoch={start_epoch}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print("\n✓ No checkpoint found. Starting fresh.")

    print("\n" + "=" * 60)
    print(f"HUNYUANDIT TRAINING: {args.epochs} epochs, {batches_per_epoch} batches/epoch")
    print(f"Mode: {args.train_mode}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr:.0e} → {_LR_ETA_MIN:.0e} (cosine, DDPM ε-prediction)")
    print("=" * 60 + "\n")

    # ── Pre-allocate reusable zero tensors (HunyuanDiT) ────────
    # zero T5/CLIP text embeddings [B, 1, cross_attn_dim]
    _cached_txt_emb = torch.zeros(
        args.batch_size, 1, model.cross_attn_dim, device=device, dtype=dtype)
    # text mask [B, 1]  — all True (attending)
    _cached_txt_mask = torch.ones(
        args.batch_size, 1, device=device, dtype=torch.bool)
    # style conditioning [B]  — 0 = default style
    _cached_style = torch.zeros(args.batch_size, device=device, dtype=torch.long)

    # ── Running stats ───────────────────────────────────────────
    _WINDOW = 100
    loss_window      = collections.deque(maxlen=_WINDOW)
    grad_norm_window = collections.deque(maxlen=_WINDOW)
    if not isinstance(ema_loss, (float, int)):
        ema_loss = None
    EMA_DECAY = 0.99

    # ── Phase-2 state ────────────────────────────────────────────
    _in_phase2         = False
    _phase2_announced  = False
    _phase2_iter       = None
    _phase2_start_step = getattr(args, 'phase2_start_step', 28801)

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    for epoch in range(start_epoch, args.epochs):
        # Update DistributedSamplers so each epoch has unique shuffling
        if world_size > 1:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for _s in diff_samplers.values():
                if _s is not None:
                    _s.set_epoch(epoch)
            if _in_phase2 and phase2_sampler is not None:
                phase2_sampler.set_epoch(epoch)
        model.transformer.train()
        epoch_losses: list[float] = []

        # Curriculum iterators
        if args.curriculum != "none":
            _diff_iters = {d: iter(diff_loaders[d]) for d in CurvtonDataset.DIFFICULTIES}
            def _next_cb(diff, _iters=_diff_iters):
                try:    return next(_iters[diff])
                except StopIteration:
                    _iters[diff] = iter(diff_loaders[diff])
                    return next(_iters[diff])
            _iter_src = range(batches_per_epoch)
        else:
            _iter_src = train_loader

        pbar = tqdm(_iter_src, desc=f"Epoch {epoch+1}/{args.epochs}")
        for _item in pbar:
            # ── Max-steps termination ────────────────────────────
            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main:
                    print(f"\n[Train] max_steps={args.max_steps} reached at step {global_step}. Stopping.")
                break

            # ── Phase-2 transition ───────────────────────────────
            if (not _in_phase2 and phase2_loader is not None
                    and global_step >= _phase2_start_step):
                _in_phase2 = True
                if not _phase2_announced and is_main:
                    print(f"\n[Phase2] Switching to triplet_dataset_train at step {global_step}")
                    _phase2_announced = True
                if phase2_sampler is not None:
                    phase2_sampler.set_epoch(epoch)
                _phase2_iter = iter(phase2_loader)

            # ── Get batch ────────────────────────────────────────
            if _in_phase2:
                try:
                    batch = next(_phase2_iter)
                except StopIteration:
                    if phase2_sampler is not None:
                        phase2_sampler.set_epoch(epoch + 100)
                    _phase2_iter = iter(phase2_loader)
                    batch = next(_phase2_iter)
            elif args.curriculum != "none":
                we, wm, wh = _curriculum_weights(global_step, args.curriculum, args.stage_steps, getattr(args, 'hard_pct', None))
                _tw = we + wm + wh
                _chosen = random.choices(
                    list(CurvtonDataset.DIFFICULTIES),
                    weights=[we/_tw, wm/_tw, wh/_tw],
                )[0]
                batch = _next_cb(_chosen)
            else:
                batch = _item

            # ── Skip incomplete/None batches ─────────────────────
            if batch is None:
                _train_log.warning(f"step={global_step}: received None batch, skipping")
                continue

            try:
                gt         = batch["ground_truth"].to(device, dtype=dtype, non_blocking=True)
                cloth      = batch["cloth"].to(device, dtype=dtype, non_blocking=True)
                person_img = batch["person"].to(
                    device, dtype=dtype, non_blocking=True)
                _train_log.debug(f"step={global_step} gt={gt.shape} cloth={cloth.shape}")

                B = gt.shape[0]

                # ── VAE encode (fused, no grad) ─────────────────────
                with torch.no_grad():
                    if args.ootd:
                        cond_input   = cloth
                        target_input = gt
                    else:
                        cond_input   = torch.cat([person_img, cloth], dim=3)
                        target_input = torch.cat([gt,         cloth], dim=3)

                    fused = torch.cat([cond_input, target_input], dim=0)
                    fused_lat = model.encode_image(fused)
                    cond_latents   = fused_lat[:B]
                    target_latents = fused_lat[B:]

                _, C_lat, H_lat, W_lat = target_latents.shape

                # ── DDPM noise (ε-prediction) ───────────────────────
                noise     = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0, model.scheduler.config.num_train_timesteps,
                    (B,), device=device, dtype=torch.long,
                )
                noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)

                # ── Channel concat: [noisy, cond] ───────────────────
                full_lat = torch.cat([noisy_latents, cond_latents], dim=1)
                H_full, W_full = full_lat.shape[2], full_lat.shape[3]

                rope_cos, rope_sin = model.get_rope_embed(H_full, W_full, device, dtype)

                px_scale = 8 if model.use_vae else 1
                meta_size = torch.tensor(
                    [[H_full * px_scale, W_full * px_scale, 0, 0, H_full * px_scale, W_full * px_scale]] * B,
                    device=device, dtype=dtype,
                )

                txt_emb   = _cached_txt_emb[:B]
                txt_mask  = _cached_txt_mask[:B]
                style_ids = _cached_style[:B]

                # ── Forward ─────────────────────────────────────────
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    if attn_score_reg is not None:
                        attn_score_reg.clear()
                    noise_pred_full = model.transformer(
                        hidden_states=full_lat,
                        timestep=timesteps,
                        encoder_hidden_states=txt_emb,
                        text_embedding_mask=txt_mask,
                        encoder_hidden_states_t5=txt_emb,
                        text_embedding_mask_t5=txt_mask,
                        image_meta_size=meta_size,
                        style=style_ids,
                        image_rotary_emb=(rope_cos, rope_sin),
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred_full[:, :C_lat]
                loss_diff = F.mse_loss(noise_pred.float(), noise.float())
                loss_attn_score_reg = torch.zeros((), device=device, dtype=loss_diff.dtype)
                if attn_score_reg is not None:
                    loss_attn_score_reg = attn_score_reg.penalty(device).to(loss_diff.dtype)
                loss = loss_diff + args.attn_score_reg_lambda * loss_attn_score_reg

                # ── Backward ────────────────────────────────────────
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.transformer.parameters(), max_norm=1.0,
                ).item()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

            except Exception as _step_exc:
                _train_log.error(f"step={global_step} FAILED — skipping:\n{traceback.format_exc()}")
                global_step += 1
                continue

            # ── Stats ───────────────────────────────────────────
            loss_val = loss.item()
            loss_window.append(loss_val)
            grad_norm_window.append(grad_norm)
            ema_loss = loss_val if ema_loss is None else EMA_DECAY * ema_loss + (1 - EMA_DECAY) * loss_val
            loss_mean = statistics.mean(loss_window)
            loss_var  = statistics.pvariance(loss_window) if len(loss_window) > 1 else 0.0
            gn_mean   = statistics.mean(grad_norm_window)
            gn_var    = statistics.pvariance(grad_norm_window) if len(grad_norm_window) > 1 else 0.0

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_losses.append(loss_val)
            wandb.log({
                "train/loss":              loss_val,
                "train/loss_diff":         loss_diff.item(),
                "train/loss_attn_score_reg": loss_attn_score_reg.item(),
                "train/loss_ema":          ema_loss,
                "train/loss_mean":         loss_mean,
                "train/loss_var":          loss_var,
                "train/grad_norm":         grad_norm,
                "train/grad_norm_mean":    gn_mean,
                "train/grad_norm_var":     gn_var,
                "train/epoch":             epoch,
                "train/timestep_mean":     timesteps.float().mean().item(),
                "train/learning_rate":     current_lr,
            }, step=global_step)

            if not _in_phase2 and args.curriculum != "none":
                _cwe, _cwm, _cwh = _curriculum_weights(global_step, args.curriculum, args.stage_steps, getattr(args, 'hard_pct', None))
                wandb.log({
                    "curriculum/w_easy":   _cwe,
                    "curriculum/w_medium": _cwm,
                    "curriculum/w_hard":   _cwh,
                    "curriculum/stage":    min(global_step // max(args.stage_steps, 1),
                                               len(_CURRIC_STAGES) - 1),
                }, step=global_step)

            wandb.log({"train/phase": 2 if _in_phase2 else 1}, step=global_step)

            # ── Image logging ───────────────────────────────────
            if global_step % args.image_log_interval == 0:
                log_images_hunyuan(global_step, batch, model,
                                   noisy_latents, noise_pred, cond_latents, target_latents,
                                   args.num_inference_steps)

            # ── Checkpoint ──────────────────────────────────────
            if global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "train_mode": args.train_mode,
                    "transformer_state_dict": model.transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "ema_loss": ema_loss,
                }, ckpt_path)
                print(f"\n💾 Saved: {ckpt_path}")

            # ── Periodic test eval ──────────────────────────────
            if (not args.skip_eval
                    and test_loaders is not None
                    and global_step > 0
                    and global_step % args.eval_interval == 0):
                print(f"\n📊 HunyuanDiT test eval at step {global_step} ...")
                eval_metrics = evaluate_on_test_hunyuan(
                    model, test_loaders, device,
                    num_inference_steps=args.num_inference_steps,
                    eval_frac=0.10,
                    ootd=args.ootd,
                )
                wandb.log(eval_metrics, step=global_step)
                print("✓ Eval metrics logged")

            pbar.set_postfix(loss=f"{loss_val:.4f}")
            global_step += 1

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            avg_loss = float('nan')
        wandb.log({"train/epoch_avg_loss": avg_loss}, step=global_step)
        print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_loss:.6f}")

        # Break outer loop if max_steps reached
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── Final save ──────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    torch.save({
        "step": global_step,
        "epoch": args.epochs,
        "train_mode": args.train_mode,
        "transformer_state_dict": model.transformer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_loss": ema_loss,
    }, final_path)

    print("\n" + "=" * 60)
    print(f"✓ HUNYUANDIT TRAINING COMPLETE!  Total steps: {global_step}")
    print(f"✓ Final checkpoint: {final_path}")
    print("=" * 60)
    wandb.finish()
    if attn_score_reg is not None:
        attn_score_reg.close()

    if world_size > 1:
        dist.destroy_process_group()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HunyuanDiT v1.1 — CatVTON Virtual Try-On Training")
    p.add_argument("--run_name", type=str, default=None,
                   help="Explicit WandB run name (also used as experiments_asset subdir). "
                        "Auto-generated from args when omitted.")
    p.add_argument("--train_mode", type=str, default="full_unet",
                   choices=["full_unet", "attention_only"],
                   help="full_unet = train entire DiT;  attention_only = freeze non-attn")
    # Dataset
    p.add_argument("--curvton_data_path", type=str, required=True)
    p.add_argument("--curvton_test_data_path", type=str, default=None)
    p.add_argument("--difficulty", type=str, default="all",
                   choices=["easy", "medium", "hard", "all"])
    p.add_argument("--gender", type=str, default="all",
                   choices=["female", "male", "all"])
    # Training
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--num_workers",  type=int,   default=32)
    p.add_argument("--curriculum",   type=str,   default="none",
                   choices=["none", "hard", "soft", "reverse", "soft_reverse"])
    p.add_argument("--stage_steps",  type=int,   default=2500)
    p.add_argument("--stage_epochs", type=int,   default=0,
                   help="Epochs per curriculum stage. If >0, overrides --stage_steps at runtime.")
    p.add_argument("--hard_pct", type=float, default=None,
                   help="Initial % of hard samples for soft_reverse/reverse curriculum.")
    p.add_argument("--data_fraction",type=float,  default=1.0)
    p.add_argument("--epochs",       type=int,   default=9999)
    p.add_argument("--lr",           type=float,  default=1e-5,
                   help="Learning rate (1e-5 recommended for DiT fine-tuning)")
    # Logging / checkpointing
    p.add_argument("--save_interval",      type=int, default=250)
    p.add_argument("--image_log_interval", type=int, default=250)
    p.add_argument("--eval_interval",      type=int, default=100)
    p.add_argument("--skip_eval", action="store_true", default=False,
                   help="Skip periodic test-set evaluation entirely")
    p.add_argument("--num_inference_steps",type=int, default=50,
                   help="UniPC steps for full inference (50 is HunyuanDiT sweet spot)")
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--no_resume",     action="store_true", default=False,
                   help="Skip auto-checkpoint detection and always train from scratch")
    p.add_argument("--checkpoint_dir",type=str, default=".")
    p.add_argument("--triplet_test_data_path", type=str, default=None,
                   help="Path to triplet_dataset root for dresscode/viton-hd test eval")
    p.add_argument("--ootd", action="store_true", default=False,
                   help="OOTDiffusion mode: cloth-only conditioning")
    p.add_argument("--max_steps",         type=int,  default=10000,
                   help="Maximum training steps (0 = run for --epochs epochs)")
    p.add_argument("--phase2_data_path",  type=str,  default=None,
                   help="Path to triplet_dataset_train for phase-2 training (hard curriculum)")
    p.add_argument("--phase2_start_step", type=int,  default=28801,
                   help="Step at which to switch from phase-1 to phase-2 data")
    p.add_argument("--use_attention_score_regularization", action="store_true", default=False,
                   help="Regularize DiT attention activation magnitude as a proxy for minimal attention.")
    p.add_argument("--attn_score_reg_lambda", type=float, default=1e-4,
                   help="Weight for attention-score proxy regularization.")

    args = p.parse_args()
    train(args)

