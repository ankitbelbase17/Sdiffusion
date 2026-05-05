"""
train_tryongan.py — TryOnGAN training loop (UNet Generator + PatchGAN Discriminator).

Conditioning: CatVTON-style spatial concatenation.
    gen_input  = cat([person, cloth], dim=width)   →  [B, 3, 512, 1024]
    gen_output = Generator(gen_input)              →  [B, 3, 512, 1024]
    tryon      = gen_output[:, :, :, :512]         →  [B, 3, 512, 512]

OOTD mode:
    gen_input  = cloth                             →  [B, 3, 512, 512]
    gen_output = Generator(gen_input)              →  [B, 3, 512, 512]
    tryon      = gen_output                        →  [B, 3, 512, 512]

Discriminator:
    D(real_or_fake_tryon, condition)  →  patch score

Losses:
    G_loss = λ_adv * GANLoss(D(fake, cond), True)
           + λ_l1  * L1(fake_tryon, gt)
           + λ_vgg * VGGPerceptualLoss(fake_tryon, gt)

    D_loss = 0.5 * [GANLoss(D(real, cond), True)
                   + GANLoss(D(fake.detach(), cond), False)]
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

_train_log = logging.getLogger("train_tryongan")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import weave

try:
    from config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, IMAGE_SIZE
except Exception:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Stable_diffusion")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))
from tryongan_model import (
    TryOnGANModel,
    GigaGANTryOnGenerator,
    MultiScaleDiscriminator,
    VGGPerceptualLoss,
    GANLoss,
    r1_penalty,
)
from utils import (
    CurvtonDataset,
    CombinedCurvtonDataset,
    collate_fn,
    get_curvton_test_dataloaders,
    get_triplet_test_dataloaders,
    get_triplet_train_loader,
    _CURRIC_STAGES,
    curriculum_weights as _curriculum_weights,
    subsample_dataset as _subsample_dataset,
)

try:
    import lpips as lpips_lib
except ImportError:
    raise ImportError("Install lpips: pip install lpips")

try:
    from torchmetrics.image import (
        FrechetInceptionDistance,
        KernelInceptionDistance,
        StructuralSimilarityIndexMeasure,
        PeakSignalNoiseRatio,
    )
except ImportError:
    raise ImportError("Install torchmetrics with image extras: pip install torchmetrics[image]")

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        raise AttributeError
    _mp_pose = mp.solutions.pose
    _POSE_AVAILABLE = True
except (ImportError, AttributeError):
    _POSE_AVAILABLE = False


# ============================================================
# HELPER: Generate from batch
# ============================================================

def _prepare_gen_input(person_img, cloth, ootd=False):
    """
    Build the Generator input tensor (channel-wise concatenation).

    Normal:  cat([person, cloth], dim=channel) → [B, 6, 512, 512]
    OOTD:    cloth                             → [B, 3, 512, 512]
    """
    if ootd:
        return cloth
    return torch.cat([person_img, cloth], dim=1)   # [B, 6, 512, 512]


def _extract_tryon(gen_output, ootd=False):
    """
    Extract the try-on prediction.
    With channel-wise concat the output is already [B, 3, 512, 512].
    """
    return gen_output


# ============================================================
# IMAGE LOGGING (GAN — no latent space)
# ============================================================

def log_images_gan(step, batch, generator, device, ootd=False):
    """Log qualitative comparison images to WandB."""
    generator.eval()
    with torch.no_grad():
        gt     = batch["ground_truth"].to(device)
        cloth  = batch["cloth"].to(device)
        person_img = batch["person"].to(device)

        gen_input  = _prepare_gen_input(person_img, cloth, ootd=ootd)
        gen_output = generator(gen_input)
        fake_tryon = _extract_tryon(gen_output, ootd=ootd)

    def to_wandb_img(tensor, caption):
        img = ((tensor[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return wandb.Image(img, caption=caption)

    gt_vis     = (gt + 1) / 2
    cloth_vis  = (cloth + 1) / 2
    _pkey      = "person" if "person" in batch else "masked_person"
    person_vis = (batch[_pkey].to(device) + 1) / 2

    wandb.log({
        "images/ground_truth":  to_wandb_img(gt, "Ground Truth"),
        "images/cloth":         to_wandb_img(cloth, "Cloth"),
        "images/person":        to_wandb_img(batch[_pkey].to(device), "Person (raw or masked)"),
        "images/generated":     to_wandb_img(fake_tryon, "Generated Try-On"),
    }, step=step)
    generator.train()


# ============================================================
# POSE KEYPOINT ERROR (reuse from utils if available)
# ============================================================

def _pose_keypoint_error(img_a_np, img_b_np):
    if not _POSE_AVAILABLE:
        return None
    try:
        H, W = img_a_np.shape[:2]
        diag = float(np.sqrt(H ** 2 + W ** 2))
        with _mp_pose.Pose(static_image_mode=True, model_complexity=1,
                           enable_segmentation=False, min_detection_confidence=0.5) as pose:
            res_a = pose.process(img_a_np)
            res_b = pose.process(img_b_np)
        if res_a.pose_landmarks is None or res_b.pose_landmarks is None:
            return None
        lm_a = res_a.pose_landmarks.landmark
        lm_b = res_b.pose_landmarks.landmark
        dists = []
        for a, b in zip(lm_a, lm_b):
            if a.visibility > 0.5 and b.visibility > 0.5:
                dx = (a.x - b.x) * W
                dy = (a.y - b.y) * H
                dists.append(np.sqrt(dx ** 2 + dy ** 2) / diag)
        return float(np.mean(dists)) if dists else None
    except Exception:
        return None


# ============================================================
# TEST-SET EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_on_test_gan(generator, test_loaders, device, eval_frac=0.10, ootd=False):
    """
    Compute LPIPS, SSIM, PSNR, FID, KID, PKE on `eval_frac` of each split.

    Generator produces pixel-space images directly (no VAE decode needed).
    """
    generator.eval()
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

    log_dict = {}
    for split_name, loader in test_loaders.items():
        n_batches = max(1, math.ceil(len(loader) * eval_frac))
        print(f"\n[Eval] {split_name}: {n_batches}/{len(loader)} batches "
              f"({eval_frac*100:.0f}% of test set) ...")

        lpips_vals, ssim_vals, psnr_vals = [], [], []
        pke_vals = []

        fid = FrechetInceptionDistance(feature=2048, reset_real_features=True,
                                       normalize=True).to(device)
        kid = KernelInceptionDistance(feature=2048, reset_real_features=True,
                                      normalize=True, subset_size=50).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            gt     = batch["ground_truth"].to(device)
            cloth  = batch["cloth"].to(device)
            person_img = batch["person"].to(device)

            gen_input  = _prepare_gen_input(person_img, cloth, ootd=ootd)
            gen_output = generator(gen_input)
            fake_tryon = _extract_tryon(gen_output, ootd=ootd)

            # Normalise to [0, 1]
            pred_01 = (fake_tryon.clamp(-1, 1) + 1) / 2     # [B,3,H,W]
            real_01 = (gt.clamp(-1, 1) + 1) / 2

            # LPIPS (expects [-1, 1])
            lp = lpips_fn(pred_01 * 2 - 1, real_01 * 2 - 1)
            lpips_vals.extend(lp.view(-1).cpu().tolist())

            # SSIM & PSNR
            ssim_vals.append(ssim_metric(pred_01, real_01).item())
            psnr_vals.append(psnr_metric(pred_01, real_01).item())

            # FID / KID
            real_u8 = (real_01 * 255).to(torch.uint8)
            pred_u8 = (pred_01 * 255).to(torch.uint8)
            fid.update(real_u8, real=True)
            fid.update(pred_u8, real=False)
            kid.update(real_u8, real=True)
            kid.update(pred_u8, real=False)

            # PKE
            if _POSE_AVAILABLE:
                for b_idx in range(real_01.shape[0]):
                    real_np = (real_01[b_idx].permute(1, 2, 0).cpu().numpy() * 255
                               ).clip(0, 255).astype(np.uint8)
                    pred_np = (pred_01[b_idx].permute(1, 2, 0).cpu().numpy() * 255
                               ).clip(0, 255).astype(np.uint8)
                    pke = _pose_keypoint_error(real_np, pred_np)
                    if pke is not None:
                        pke_vals.append(pke)

        fid_score         = fid.compute().item()
        kid_mean, kid_std = kid.compute()

        lpips_arr = np.array(lpips_vals) if lpips_vals else np.zeros(1)
        ssim_arr  = np.array(ssim_vals) if ssim_vals else np.zeros(1)
        psnr_arr  = np.array(psnr_vals) if psnr_vals else np.zeros(1)

        log_dict[f"test/{split_name}/lpips_mean"] = float(lpips_arr.mean())
        log_dict[f"test/{split_name}/lpips_std"]  = float(lpips_arr.std(ddof=1) if len(lpips_arr) > 1 else 0.0)
        log_dict[f"test/{split_name}/ssim_mean"]  = float(ssim_arr.mean())
        log_dict[f"test/{split_name}/ssim_std"]   = float(ssim_arr.std(ddof=1) if len(ssim_arr) > 1 else 0.0)
        log_dict[f"test/{split_name}/psnr_mean"]  = float(psnr_arr.mean())
        log_dict[f"test/{split_name}/psnr_std"]   = float(psnr_arr.std(ddof=1) if len(psnr_arr) > 1 else 0.0)
        log_dict[f"test/{split_name}/fid"]        = fid_score
        log_dict[f"test/{split_name}/kid_mean"]   = kid_mean.item()
        log_dict[f"test/{split_name}/kid_std"]     = kid_std.item()

        if pke_vals:
            pke_arr = np.array(pke_vals)
            log_dict[f"test/{split_name}/pke_mean"] = float(pke_arr.mean())
            log_dict[f"test/{split_name}/pke_std"]  = float(pke_arr.std(ddof=1) if len(pke_arr) > 1 else 0.0)
            pke_str = (f"  PKE={log_dict[f'test/{split_name}/pke_mean']:.4f}"
                       f"±{log_dict[f'test/{split_name}/pke_std']:.4f}")
        else:
            pke_str = "  PKE=N/A"

        print(f"   {split_name:6s} | "
              f"LPIPS={log_dict[f'test/{split_name}/lpips_mean']:.4f}±{log_dict[f'test/{split_name}/lpips_std']:.4f}  "
              f"SSIM={log_dict[f'test/{split_name}/ssim_mean']:.4f}±{log_dict[f'test/{split_name}/ssim_std']:.4f}  "
              f"PSNR={log_dict[f'test/{split_name}/psnr_mean']:.2f}±{log_dict[f'test/{split_name}/psnr_std']:.2f} dB  "
              f"FID={fid_score:.2f}  "
              f"KID={kid_mean.item():.4f}±{kid_std.item():.4f}"
              + pke_str)

        del fid, kid, ssim_metric, psnr_metric
        torch.cuda.empty_cache()

    del lpips_fn
    torch.cuda.empty_cache()
    generator.train()
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

    if is_main:
        wandb.login(key=WANDB_API_KEY)
        weave.init(f"{os.getenv('WANDB_ENTITY', WANDB_ENTITY)}/{os.getenv('WANDB_PROJECT', WANDB_PROJECT)}")
        print(f"Device: {device}  |  rank={rank}/{world_size}")

    # ── GPU performance flags ───────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if is_main:
            print("✓ cudnn.benchmark=True, TF32 enabled")

    # ── Model ───────────────────────────────────────────────────
    g_in_ch = 3 if args.ootd else 6
    d_in_ch = 6 if args.ootd else 9
    gan = TryOnGANModel(in_channels_g=g_in_ch, in_channels_d=d_in_ch,
                        style_dim=args.style_dim,
                        n_kernels=args.n_kernels,
                        n_disc_scales=args.n_disc_scales).to(device)
    G = gan.generator
    D = gan.discriminator

    vgg_loss = VGGPerceptualLoss().to(device)
    gan_loss = GANLoss().to(device)

    # OOTD mode banner
    if args.ootd:
        print("\n🔶 OOTDiffusion mode: cloth-only conditioning (no person)")

    # torch.compile
    if hasattr(torch, "compile"):
        if is_main:
            print("\n⚙  Compiling Generator with torch.compile (mode='reduce-overhead') ...")
        G = torch.compile(G, mode="reduce-overhead")
        if is_main:
            print("   ✓ torch.compile applied to Generator")

    # ── DDP Wrap ────────────────────────────────────────────────
    if world_size > 1:
        G = DDP(G, device_ids=[local_rank], output_device=local_rank)
        D = DDP(D, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"✓ DDP: {world_size} GPUs, total batch={args.batch_size * world_size}")

    _frac_tag   = f"_frac{int(args.data_fraction * 100)}pct" if args.data_fraction < 1.0 else ""
    _phase2_tag = "_phase2" if getattr(args, 'phase2_data_path', None) else ""
    _ootd_tag   = "_ootd"   if args.ootd else ""
    _auto_name  = (
        f"tryongan"
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

    # ── Experiments directory ────────────────────────────────────
    _EXPERIMENTS_ROOT = "/iopsstor/scratch/cscs/dbartaula/experiments_assets"
    experiments_dir = os.path.join(_EXPERIMENTS_ROOT, run_name)
    ckpt_dir = os.path.join(experiments_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"✓ Checkpoint dir: {ckpt_dir}")

    # ── Datasets & DataLoaders ──────────────────────────────────
    genders = ("female", "male") if args.gender == "all" else (args.gender,)

    diff_loaders  = {}
    diff_samplers = {}

    if args.dataset == "triplet":
        # ── Triplet-only mode: train on vitonhd + dresscode mixture ──
        if not args.triplet_train_data_path:
            raise ValueError("--triplet_train_data_path is required when --dataset=triplet")
        train_loader, train_sampler = get_triplet_train_loader(
            root_dir=args.triplet_train_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
            world_size=world_size,
            rank=rank,
        )
        dataset_label = f"Triplet-{args.curriculum}"
        batches_per_epoch = len(train_loader)
        if is_main:
            print(f"✓ Triplet DataLoader: {batches_per_epoch} batches per epoch")
    else:
        # ── CurvTon mode (default) ──
        if not args.curvton_data_path:
            raise ValueError("--curvton_data_path is required when --dataset=curvton")
        for _diff in CurvtonDataset.DIFFICULTIES:
            _ds = CombinedCurvtonDataset(
                root_dir=args.curvton_data_path,
                difficulties=(_diff,),
                genders=genders,
                size=IMAGE_SIZE,
            )
            _ds = _subsample_dataset(_ds, args.data_fraction)
            _sampler = DistributedSampler(_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
            diff_samplers[_diff] = _sampler
            diff_loaders[_diff] = DataLoader(
                _ds,
                batch_size=args.batch_size,
                shuffle=(_sampler is None),
                sampler=_sampler,
                num_workers=args.num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=(4 if args.num_workers > 0 else None),
            )
            if is_main:
                print(f"✓ CurvTon [{_diff}] DataLoader: {len(diff_loaders[_diff])} batches/epoch")

        if args.difficulty == "all":
            difficulties = CurvtonDataset.DIFFICULTIES
        else:
            difficulties = (args.difficulty,)
        train_dataset = CombinedCurvtonDataset(
            root_dir=args.curvton_data_path,
            difficulties=difficulties,
            genders=genders,
            size=IMAGE_SIZE,
        )
        train_dataset = _subsample_dataset(train_dataset, args.data_fraction)
        _frac_tag = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
        dataset_label = f"CurvTon-{args.difficulty}-{args.gender}-{args.curriculum}{_frac_tag}"

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
        )
        batches_per_epoch = len(train_loader)
        if is_main:
            print(f"✓ DataLoader ({dataset_label}): {batches_per_epoch} batches per epoch")

    # ── Test loaders for periodic evaluation ────────────────────
    test_loaders: dict = {}
    if args.curvton_test_data_path:
        genders_test = ("female", "male") if args.gender == "all" else (args.gender,)
        if is_main:
            print(f"\nBuilding CurvTon test loaders from {args.curvton_test_data_path} ...")
        for k, v in get_curvton_test_dataloaders(
            root_dir=args.curvton_test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
            genders=genders_test,
        ).items():
            test_loaders[f"curvton_{k}"] = v
        if is_main:
            print(f"✓ CurvTon test loaders: {[k for k in test_loaders]}")
    elif is_main:
        print("\n[Eval] No --curvton_test_data_path provided; CurvTon test eval skipped.")
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
    elif is_main:
        print("[Eval] No --triplet_test_data_path; triplet eval skipped.")
    if not test_loaders:
        test_loaders = None

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

    # Count trainable params
    _G_raw = G.module if isinstance(G, DDP) else G
    _D_raw = D.module if isinstance(D, DDP) else D
    g_trainable = sum(p.numel() for p in _G_raw.parameters() if p.requires_grad)
    d_trainable = sum(p.numel() for p in _D_raw.parameters() if p.requires_grad)
    if is_main:
        print(f"\nTrainable params:  G={g_trainable:,}  D={d_trainable:,}  "
              f"total={g_trainable + d_trainable:,}")

    # ── WandB ───────────────────────────────────────────────────
    if is_main:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", WANDB_PROJECT),
            entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
            id=run_name,
            resume="allow",
            config={
                "arch": "GigaGAN-TryOnGAN",
                "lr_g": args.lr_g,
                "lr_d": args.lr_d,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "style_dim": args.style_dim,
                "n_kernels": args.n_kernels,
                "n_disc_scales": args.n_disc_scales,
                "lambda_adv": args.lambda_adv,
                "lambda_l1": args.lambda_l1,
                "lambda_vgg": args.lambda_vgg,
                "lambda_r1": args.lambda_r1,
                "r1_interval": args.r1_interval,
                "curriculum": args.curriculum,
                "stage_steps": args.stage_steps,
                "data_fraction": args.data_fraction,
                "ootd": args.ootd,
                "dataset": dataset_label,
                "g_trainable": g_trainable,
                "d_trainable": d_trainable,
                "max_steps": args.max_steps,
                "phase2_data_path": getattr(args, 'phase2_data_path', None),
                "phase2_start_step": getattr(args, 'phase2_start_step', 28801),
            },
            name=run_name,
        )

    # ── Optimizers ──────────────────────────────────────────────
    optimizer_G = Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    _total_steps = (
        args.max_steps if args.max_steps > 0
        else args.epochs * batches_per_epoch
    )
    _LR_ETA_MIN  = 1e-6
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=max(_total_steps, 1), eta_min=_LR_ETA_MIN)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=max(_total_steps, 1), eta_min=_LR_ETA_MIN)

    global_step = 0
    start_epoch = 0
    ema_loss_g  = None
    ema_loss_d  = None

    # ── Checkpoint Resume ───────────────────────────────────────
    ckpt_to_load = None
    if args.resume:
        ckpt_to_load = args.resume
        if is_main:
            print(f"\n▶ --resume specified: {ckpt_to_load}")
    elif not args.no_resume:
        candidates = (
            glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt"))
            + glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_tryongan_step_*.pt"))
        )
        if candidates:
            def _step_num(p):
                try:
                    return int(os.path.basename(p).split("_step_")[1].replace(".pt", ""))
                except Exception:
                    return -1
            ckpt_to_load = max(candidates, key=_step_num)
            if is_main:
                print(f"\n▶ Auto-detected latest checkpoint: {ckpt_to_load}")
    else:
        if is_main:
            print("\n▶ --no_resume set: training from scratch.")

    if ckpt_to_load:
        if is_main:
            print(f"   Loading checkpoint …")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        _G_ckpt = G.module if isinstance(G, DDP) else G
        _D_ckpt = D.module if isinstance(D, DDP) else D
        _G_ckpt.load_state_dict(ckpt["g_state_dict"])
        _D_ckpt.load_state_dict(ckpt["d_state_dict"])
        optimizer_G.load_state_dict(ckpt["optimizer_g_state_dict"])
        optimizer_D.load_state_dict(ckpt["optimizer_d_state_dict"])
        global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        if "scheduler_g_state_dict" in ckpt:
            scheduler_G.load_state_dict(ckpt["scheduler_g_state_dict"])
            scheduler_D.load_state_dict(ckpt["scheduler_d_state_dict"])
            if is_main:
                print(f"   ✓ LR schedulers restored.")
        else:
            for _ in range(global_step):
                scheduler_G.step()
                scheduler_D.step()
            if is_main:
                print(f"   ⚠ No scheduler state — fast-forwarded {global_step} steps.")
        if "scaler_g_state_dict" in ckpt:
            scaler_G.load_state_dict(ckpt["scaler_g_state_dict"])
            scaler_D.load_state_dict(ckpt["scaler_d_state_dict"])
            if is_main:
                print(f"   ✓ GradScaler states restored.")
        if "ema_loss_g" in ckpt and ckpt["ema_loss_g"] is not None:
            ema_loss_g = ckpt["ema_loss_g"]
        if "ema_loss_d" in ckpt and ckpt["ema_loss_d"] is not None:
            ema_loss_d = ckpt["ema_loss_d"]
        if is_main:
            print(f"   ✓ Resumed — step={global_step}, epoch={start_epoch}")
    elif is_main:
        print("\n✓ No checkpoint found. Starting fresh.")

    if is_main:
        print("\n" + "=" * 60)
        print(f"TRAINING: {args.epochs} epochs, {batches_per_epoch} batches/epoch")
        print(f"G lr={args.lr_g}, D lr={args.lr_d}, batch_size={args.batch_size}")
        print(f"Losses: λ_adv={args.lambda_adv}, λ_l1={args.lambda_l1}, λ_vgg={args.lambda_vgg}")
        print(f"LR Schedule: cosine anneal → {_LR_ETA_MIN:.0e} over {_total_steps} steps")
        print("=" * 60 + "\n")

    # ── Running stats ───────────────────────────────────────────
    _WINDOW  = 100
    EMA_DECAY = 0.99
    loss_g_window   = collections.deque(maxlen=_WINDOW)
    loss_d_window   = collections.deque(maxlen=_WINDOW)
    grad_norm_g_win = collections.deque(maxlen=_WINDOW)
    grad_norm_d_win = collections.deque(maxlen=_WINDOW)
    # ── Phase-2 state ─────────────────────────────────────────
    _in_phase2         = False
    _phase2_announced  = False
    _phase2_iter       = None
    _phase2_start_step = getattr(args, 'phase2_start_step', 28801)
    # ── Training loop ───────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for _s in diff_samplers.values():
                if _s is not None:
                    _s.set_epoch(epoch)
        G.train()
        D.train()
        epoch_losses_g = []
        epoch_losses_d = []

        # Curriculum iterators (reset per epoch)
        if args.curriculum != "none":
            _diff_iters = {d: iter(diff_loaders[d]) for d in CurvtonDataset.DIFFICULTIES}

            def _next_curriculum_batch(diff, _iters=_diff_iters):
                try:
                    return next(_iters[diff])
                except StopIteration:
                    _iters[diff] = iter(diff_loaders[diff])
                    return next(_iters[diff])

            _iter_src = range(batches_per_epoch)
        else:
            _iter_src = train_loader

        pbar = tqdm(_iter_src, desc=f"Epoch {epoch+1}/{args.epochs}")
        for _item in pbar:
            # Curriculum sampling
            if args.curriculum != "none":
                we, wm, wh = _curriculum_weights(global_step, args.curriculum, args.stage_steps, getattr(args, 'hard_pct', None))
                _tw = we + wm + wh
                _chosen = random.choices(
                    list(CurvtonDataset.DIFFICULTIES),
                    weights=[we / _tw, wm / _tw, wh / _tw],
                )[0]
                batch = _next_curriculum_batch(_chosen)
            else:
                batch = _item

            gt         = batch["ground_truth"].to(device, non_blocking=True)
            cloth      = batch["cloth"].to(device, non_blocking=True)
            person_img = batch["person"].to(device, non_blocking=True)
            B = gt.shape[0]

            gen_input = _prepare_gen_input(person_img, cloth, ootd=args.ootd)
            cond = gen_input   # condition for discriminator

            # ════════════════════════════════════════════════════
            # 1) UPDATE DISCRIMINATOR
            # ════════════════════════════════════════════════════
            optimizer_D.zero_grad(set_to_none=True)
            with autocast():
                # Real
                real_tryon = gt  # [B, 3, 512, 512]
                d_real = D(real_tryon, cond)
                loss_d_real = gan_loss(d_real, target_is_real=True)

                # Fake
                with torch.no_grad():
                    gen_output = G(gen_input)
                fake_tryon = _extract_tryon(gen_output, ootd=args.ootd)
                d_fake = D(fake_tryon.detach(), cond)
                loss_d_fake = gan_loss(d_fake, target_is_real=False)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            # R1 gradient penalty (lazy regularisation)
            loss_r1 = torch.tensor(0.0, device=device)
            if args.lambda_r1 > 0 and global_step % args.r1_interval == 0:
                real_tryon_r1 = real_tryon.detach().requires_grad_(True)
                d_real_r1 = D(real_tryon_r1, cond.detach())
                loss_r1 = r1_penalty(d_real_r1, real_tryon_r1)
                # Scale by interval (lazy reg) and weight
                loss_d = loss_d + (args.lambda_r1 / 2) * loss_r1 * args.r1_interval

            scaler_D.scale(loss_d).backward()
            scaler_D.unscale_(optimizer_D)
            grad_norm_d = nn.utils.clip_grad_norm_(D.parameters(), max_norm=1e9).item()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # ════════════════════════════════════════════════════
            # 2) UPDATE GENERATOR
            # ════════════════════════════════════════════════════
            optimizer_G.zero_grad(set_to_none=True)
            with autocast():
                gen_output = G(gen_input)
                fake_tryon = _extract_tryon(gen_output, ootd=args.ootd)

                # Adversarial
                d_fake_for_g = D(fake_tryon, cond)
                loss_g_adv = gan_loss(d_fake_for_g, target_is_real=True)

                # Pixel L1
                loss_g_l1 = F.l1_loss(fake_tryon, gt)

                # VGG Perceptual
                loss_g_vgg = vgg_loss(fake_tryon, gt)

                loss_g = (args.lambda_adv * loss_g_adv
                          + args.lambda_l1 * loss_g_l1
                          + args.lambda_vgg * loss_g_vgg)

            scaler_G.scale(loss_g).backward()
            scaler_G.unscale_(optimizer_G)
            grad_norm_g = nn.utils.clip_grad_norm_(G.parameters(), max_norm=1e9).item()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            scheduler_G.step()
            scheduler_D.step()

            # ── Running stats ───────────────────────────────────
            lg = loss_g.item()
            ld = loss_d.item()
            loss_g_window.append(lg)
            loss_d_window.append(ld)
            grad_norm_g_win.append(grad_norm_g)
            grad_norm_d_win.append(grad_norm_d)
            epoch_losses_g.append(lg)
            epoch_losses_d.append(ld)

            ema_loss_g = lg if ema_loss_g is None else EMA_DECAY * ema_loss_g + (1 - EMA_DECAY) * lg
            ema_loss_d = ld if ema_loss_d is None else EMA_DECAY * ema_loss_d + (1 - EMA_DECAY) * ld

            if is_main:
                wandb.log({
                    "train/loss_g":            lg,
                    "train/loss_g_adv":        loss_g_adv.item(),
                    "train/loss_g_l1":         loss_g_l1.item(),
                    "train/loss_g_vgg":        loss_g_vgg.item(),
                    "train/loss_d":            ld,
                    "train/loss_d_real":       loss_d_real.item(),
                    "train/loss_d_fake":       loss_d_fake.item(),
                    "train/loss_d_r1":         loss_r1.item(),
                    "train/loss_g_ema":        ema_loss_g,
                    "train/loss_d_ema":        ema_loss_d,
                    "train/loss_g_mean":       statistics.mean(loss_g_window),
                    "train/loss_d_mean":       statistics.mean(loss_d_window),
                    "train/grad_norm_g":       grad_norm_g,
                    "train/grad_norm_d":       grad_norm_d,
                    "train/learning_rate_g":   optimizer_G.param_groups[0]["lr"],
                    "train/learning_rate_d":   optimizer_D.param_groups[0]["lr"],
                    "train/epoch":             epoch,
                    "train/phase":             2 if _in_phase2 else 1,
                }, step=global_step)

                # Curriculum logging
                if not _in_phase2 and args.curriculum != "none":
                    _cwe, _cwm, _cwh = _curriculum_weights(global_step, args.curriculum, args.stage_steps, getattr(args, 'hard_pct', None))
                    wandb.log({
                        "curriculum/w_easy":   _cwe,
                        "curriculum/w_medium": _cwm,
                        "curriculum/w_hard":   _cwh,
                        "curriculum/stage":    min(global_step // max(args.stage_steps, 1),
                                                   len(_CURRIC_STAGES) - 1),
                    }, step=global_step)

                # Log images
                if global_step % args.image_log_interval == 0:
                    _G_log = G.module if isinstance(G, DDP) else G
                    log_images_gan(global_step, batch, _G_log, device, ootd=args.ootd)

                # Save checkpoint
                if global_step > 0 and global_step % args.save_interval == 0:
                    _G_save = G.module if isinstance(G, DDP) else G
                    _D_save = D.module if isinstance(D, DDP) else D
                    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "g_state_dict": _G_save.state_dict(),
                        "d_state_dict": _D_save.state_dict(),
                        "optimizer_g_state_dict": optimizer_G.state_dict(),
                        "optimizer_d_state_dict": optimizer_D.state_dict(),
                        "scheduler_g_state_dict": scheduler_G.state_dict(),
                        "scheduler_d_state_dict": scheduler_D.state_dict(),
                        "scaler_g_state_dict": scaler_G.state_dict(),
                        "scaler_d_state_dict": scaler_D.state_dict(),
                        "ema_loss_g": ema_loss_g,
                        "ema_loss_d": ema_loss_d,
                    }, ckpt_path)
                    print(f"\n💾 Saved: {ckpt_path}")

                # Periodic test-set evaluation
                if (not args.skip_eval
                        and test_loaders is not None
                        and global_step > 0
                        and global_step % args.eval_interval == 0):
                    _G_eval = G.module if isinstance(G, DDP) else G
                    print(f"\n📊 Running test-set evaluation at step {global_step} ...")
                    eval_metrics = evaluate_on_test_gan(
                        _G_eval, test_loaders, device,
                        eval_frac=0.10, ootd=args.ootd,
                    )
                    wandb.log(eval_metrics, step=global_step)
                    print(f"✓ Eval metrics logged to W&B")

            pbar.set_postfix(G=f"{lg:.4f}", D=f"{ld:.4f}")
            global_step += 1

        if epoch_losses_g:
            avg_g = sum(epoch_losses_g) / len(epoch_losses_g)
        else:
            avg_g = float('nan')
        if epoch_losses_d:
            avg_d = sum(epoch_losses_d) / len(epoch_losses_d)
        else:
            avg_d = float('nan')
        if is_main:
            wandb.log({"train/epoch_avg_loss_g": avg_g, "train/epoch_avg_loss_d": avg_d},
                      step=global_step)
            print(f"\nEpoch {epoch+1} complete.  Avg G loss: {avg_g:.6f}  Avg D loss: {avg_d:.6f}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save (rank-0 only)
    if is_main:
        _G_final = G.module if isinstance(G, DDP) else G
        _D_final = D.module if isinstance(D, DDP) else D
        final_path = os.path.join(ckpt_dir, "ckpt_final.pt")
        torch.save({
            "step": global_step,
            "epoch": args.epochs,
            "g_state_dict": _G_final.state_dict(),
            "d_state_dict": _D_final.state_dict(),
            "optimizer_g_state_dict": optimizer_G.state_dict(),
            "optimizer_d_state_dict": optimizer_D.state_dict(),
            "scheduler_g_state_dict": scheduler_G.state_dict(),
            "scheduler_d_state_dict": scheduler_D.state_dict(),
            "scaler_g_state_dict": scaler_G.state_dict(),
            "scaler_d_state_dict": scaler_D.state_dict(),
            "ema_loss_g": ema_loss_g,
            "ema_loss_d": ema_loss_d,
        }, final_path)
        print("\n" + "=" * 60)
        print(f"✓ TRAINING COMPLETE! Total steps: {global_step}")
        print(f"✓ Final checkpoint: {final_path}")
        print("=" * 60)
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TryOnGAN Training (CurvTon)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="curvton",
                        choices=["curvton", "triplet"],
                        help="Dataset to train on: curvton (default) or triplet (vitonhd+dresscode)")
    parser.add_argument("--curvton_data_path", type=str, default=None,
                        help="S3 bucket or path to CurvTon dataset (e.g. curvton-dataset)")
    parser.add_argument("--triplet_train_data_path", type=str, default=None,
                        help="Path to triplet_dataset_train root for vitonhd+dresscode training")
    parser.add_argument("--curvton_test_data_path", type=str, default=None,
                        help="S3 bucket or path to CurvTon test dataset")
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--gender", type=str, default="all",
                        choices=["female", "male", "all"])

    # Architecture
    parser.add_argument("--style_dim", type=int, default=512,
                        help="GigaGAN style vector dimensionality (default: 512)")
    parser.add_argument("--n_kernels", type=int, default=4,
                        help="Number of basis kernels for adaptive kernel selection (default: 4)")
    parser.add_argument("--n_disc_scales", type=int, default=3,
                        help="Number of discriminator scales (default: 3)")

    # Loss weights
    parser.add_argument("--lambda_adv", type=float, default=1.0,
                        help="Weight for adversarial loss (default: 1.0)")
    parser.add_argument("--lambda_l1", type=float, default=10.0,
                        help="Weight for L1 pixel loss (default: 10.0)")
    parser.add_argument("--lambda_vgg", type=float, default=10.0,
                        help="Weight for VGG perceptual loss (default: 10.0)")
    parser.add_argument("--lambda_r1", type=float, default=10.0,
                        help="Weight for R1 gradient penalty (default: 10.0)")
    parser.add_argument("--r1_interval", type=int, default=16,
                        help="Apply R1 penalty every N D steps (lazy regularisation, default: 16)")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_g", type=float, default=2e-4,
                        help="Generator learning rate (default: 2e-4)")
    parser.add_argument("--lr_d", type=float, default=2e-4,
                        help="Discriminator learning rate (default: 2e-4)")

    # Curriculum
    parser.add_argument("--run_name", type=str, default=None,
                        help="Explicit WandB run name (also used as experiments_asset subdir). "
                             "Auto-generated from args when omitted.")
    parser.add_argument("--curriculum", type=str, default="none",
                        choices=["none", "hard", "soft", "reverse", "soft_reverse"])
    parser.add_argument("--stage_steps", type=int, default=2500,
                        help="Steps per curriculum stage (default: 2500)")
    parser.add_argument("--hard_pct", type=float, default=None,
                        help="Initial % of hard samples for soft_reverse/reverse curriculum.")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (default: 1.0)")

    # Logging / Checkpointing
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--image_log_interval", type=int, default=250)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--skip_eval", action="store_true", default=False,
                        help="Skip periodic test-set evaluation entirely")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    parser.add_argument("--no_resume", action="store_true", default=False,
                        help="Skip auto-checkpoint detection and always train from scratch")
    parser.add_argument("--checkpoint_dir", type=str, default=".",
                        help="Directory to scan for latest checkpoint")
    parser.add_argument("--triplet_test_data_path", type=str, default=None,
                        help="Path to triplet_dataset root for dresscode/viton-hd test eval")

    # OOTD mode
    parser.add_argument("--ootd", action="store_true", default=False,
                        help="OOTDiffusion mode: cloth-only conditioning (no person)")
    parser.add_argument("--max_steps",         type=int,  default=10000,
                        help="Maximum training steps (0 = run for --epochs epochs)")
    parser.add_argument("--phase2_data_path",  type=str,  default=None,
                        help="Path to triplet_dataset_train for phase-2 training (hard curriculum)")
    parser.add_argument("--phase2_start_step", type=int,  default=28801,
                        help="Step at which to switch from phase-1 to phase-2 data")

    args = parser.parse_args()
    train(args)

