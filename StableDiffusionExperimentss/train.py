"""
train.py — Main training loop and CLI entry-point.
"""

import os
import glob
import collections
import statistics
import random
import argparse
import traceback
import logging
import datetime
import time

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
_train_log = logging.getLogger("train")
# Suppress noisy PIL/Pillow chunk-level debug output
logging.getLogger("PIL").setLevel(logging.WARNING)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.amp import GradScaler as _GradScaler
    def GradScaler(): return _GradScaler('cuda')     # PyTorch >= 1.10
except ImportError:
    from torch.cuda.amp import GradScaler            # PyTorch < 1.10
try:
    from torch.amp import autocast as _amp_autocast
    def autocast(): return _amp_autocast('cuda')     # PyTorch >= 1.10
except ImportError:
    from torch.cuda.amp import autocast              # PyTorch < 1.10
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
try:
    import wandb
    _WANDB_IMPORT_ERROR = None
except Exception as _e:
    wandb = None
    _WANDB_IMPORT_ERROR = _e
try:
    import weave
except Exception:
    weave = None

try:
    from config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, MODEL_NAME, IMAGE_SIZE as CFG_IMAGE_SIZE
except Exception:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Stable_diffusion")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
    CFG_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))
from model import SDModel, freeze_non_attention, print_trainable_params
from utils import (
    VitonHDDataset,
    CurvtonDataset,
    CombinedCurvtonDataset,
    collate_fn,
    get_curvton_test_dataloaders,
    get_triplet_test_dataloaders,
    get_triplet_train_loader,
    decode_latents,
    run_full_inference,
    log_images,
    log_images_distributed,
    evaluate_on_test,
    _CURRIC_STAGES,
    curriculum_weights as _curriculum_weights,
    subsample_dataset as _subsample_dataset,
)

class _AttentionActivationRegularizer:
    """Collects attention-module output magnitudes as a proxy for attention intensity."""

    def __init__(self, unet, include_self=True, include_cross=True):
        self._handles = []
        self._vals = []
        for name, module in unet.named_modules():
            if not hasattr(module, "to_q"):
                continue
            is_self = ".attn1" in name
            is_cross = ".attn2" in name
            if (is_self and not include_self) or (is_cross and not include_cross):
                continue
            if not (is_self or is_cross):
                continue
            self._handles.append(module.register_forward_hook(self._hook))

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


# ============================================================
# MAIN TRAINING
# ============================================================
def train(args):
    image_size = args.image_size
    # ── Distributed Data Parallel setup ────────────────────────
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
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

    wandb_enabled = False
    if is_main:
        # Allow training to continue even if wandb is unavailable/broken on cluster envs.
        if os.environ.get("DISABLE_WANDB", "0") == "1":
            print("[INFO] W&B disabled via DISABLE_WANDB=1")
        elif wandb is None:
            print(f"[WARN] W&B unavailable, disabling logging: {_WANDB_IMPORT_ERROR}")
        else:
            try:
                wandb.login(key=WANDB_API_KEY)
                if weave is not None:
                    weave.init(f"{os.getenv('WANDB_ENTITY', WANDB_ENTITY)}/{os.getenv('WANDB_PROJECT', WANDB_PROJECT)}")
                wandb_enabled = True
            except Exception as _we:
                print(f"[WARN] W&B init failed, disabling logging: {_we}")
        print(f"Device: {device}  |  rank={rank}/{world_size}")
        if args.use_dream:
            print(f"[DREAM] Enabled with lambda={args.dream_lambda}")
        if args.use_attention_regularization:
            print(
                f"[ATTN-REG] Enabled with lambda={args.attn_reg_lambda}, "
                f"threshold={args.attn_reg_threshold}"
            )
        if args.image_size <= 0:
            print("[INFO] image_size<=0: no resize mode enabled (uses native dataset resolution).")
        else:
            print(f"[INFO] image_size={args.image_size}")

    def _wandb_log(payload, step=None):
        if is_main and wandb_enabled:
            try:
                wandb.log(payload, step=step)
            except Exception as _log_e:
                _train_log.warning("wandb.log failed at step=%s: %s", step, _log_e)

    # ── GPU performance flags ───────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True          # auto-tune convolution kernels
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on Ampere+ for matmul
        torch.backends.cudnn.allow_tf32 = True          # TF32 for cuDNN convolutions
        if is_main:
            print("✓ cudnn.benchmark=True, TF32 enabled")

    # Decide dataset
    use_curvton = (args.dataset == "curvton")
    if use_curvton:
        print(f"Data source: CurvTon from {args.curvton_data_path} "
              f"(difficulty={args.difficulty}, gender={args.gender})")
    else:
        print(f"Data source: VITON-HD from {args.viton_data_path}")

    # Model
    model = SDModel().to(device)

    # OOTD mode banner
    if args.ootd:
        print("\n🔶 OOTDiffusion mode: cloth-only conditioning (no person/mask in cond)")

    # Set trainable parameters based on mode
    if args.train_mode == "attention_only":
        model.unet = freeze_non_attention(model.unet)
        _mode_tag = "attn"
    else:
        _mode_tag = "full"

    attn_score_reg = None
    if args.use_attention_score_regularization:
        attn_score_reg = _AttentionActivationRegularizer(
            model.unet,
            include_self=True,
            include_cross=True,
        )

    _frac_tag   = f"_frac{int(args.data_fraction * 100)}pct" if args.data_fraction < 1.0 else ""
    _phase2_tag = "_phase2" if getattr(args, 'phase2_data_path', None) else ""
    _ootd_tag   = "_ootd"   if args.ootd else ""
    _auto_name  = (
        f"{args.dataset}_sd_{_mode_tag}"
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

    # Print trainable parameters
    trainable_params = print_trainable_params(model, args.train_mode)

    # torch.compile (PyTorch 2.0+) — speeds up training via graph compilation
    # Use backend='eager' to avoid Triton dependency (inductor requires Triton)
    if hasattr(torch, "compile"):
        if is_main:
            print("\n⚙  Compiling UNet with torch.compile (backend='eager') ...")
        model.unet = torch.compile(model.unet, backend="eager")
        if is_main:
            print("   ✓ torch.compile applied to UNet (eager backend, no Triton required).")
    else:
        if is_main:
            print("\n⚠  torch.compile not available (requires PyTorch >= 2.0). Skipping.")

    # ── DDP Wrap (after compile, before DataLoaders) ─────────────────
    if world_size > 1:
        model.unet = DDP(model.unet, device_ids=[local_rank],
                         output_device=local_rank, find_unused_parameters=False)
        if is_main:
            print(f"✓ DDP: {world_size} GPUs, total batch={args.batch_size * world_size}")

    # Dataset & DataLoader
    diff_loaders: dict = {}   # per-difficulty loaders — built for curriculum sampling
    diff_samplers: dict = {}  # DistributedSamplers for set_epoch
    if use_curvton:
        genders = ("female", "male") if args.gender == "all" else (args.gender,)

        # Build one DataLoader per difficulty (needed regardless of curriculum mode)
        for _diff in CurvtonDataset.DIFFICULTIES:
            _ds = CombinedCurvtonDataset(
                root_dir=args.curvton_data_path,
                difficulties=(_diff,),
                genders=genders,
                size=image_size,
            )
            _ds = _subsample_dataset(_ds, args.data_fraction)
            _samp = (DistributedSampler(_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True, drop_last=True)
                     if world_size > 1 else None)
            diff_samplers[_diff] = _samp
            diff_loaders[_diff] = DataLoader(
                _ds,
                batch_size=args.batch_size,
                shuffle=(_samp is None),
                sampler=_samp,
                num_workers=args.num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=(4 if args.num_workers > 0 else None),
            )
            if is_main:
                print(f"✓ CurvTon [{_diff}] DataLoader: {len(diff_loaders[_diff])} batches/epoch")

        # "All" combined loader — used for non-curriculum and batches_per_epoch reference
        difficulty_map = {
            "all": CurvtonDataset.DIFFICULTIES,
            "easy_hard": ("easy", "hard"),
            "medium_hard": ("medium", "hard"),
        }
        difficulties = difficulty_map.get(args.difficulty, (args.difficulty,))
        train_dataset = CombinedCurvtonDataset(
            root_dir=args.curvton_data_path,
            difficulties=difficulties,
            genders=genders,
            size=image_size,
        )
        train_dataset = _subsample_dataset(train_dataset, args.data_fraction)
        _frac_tag = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
        dataset_label = f"CurvTon-{args.difficulty}-{args.gender}-{args.curriculum}{_frac_tag}"
    else:
        train_dataset = VitonHDDataset(
            root_dir=args.viton_data_path,
            split='train',
            size=image_size,
        )
        train_dataset = _subsample_dataset(train_dataset, args.data_fraction)
        _frac_tag = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
        dataset_label = f"VITON-HD{_frac_tag}"

    _train_sampler = (DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                          shuffle=True, drop_last=True)
                       if world_size > 1 else None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(_train_sampler is None),
        sampler=_train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
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
        print(f"✓ DataLoader ({dataset_label}): {batches_per_epoch} batches/epoch per GPU")

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
                size=image_size,
                world_size=world_size,
                rank=rank,
            )
            if is_main:
                print(f"[Phase2] ✓ loader ready: {len(phase2_loader)} batches/epoch")
        except Exception:
            if is_main:
                print(f"[Phase2] WARNING: could not build phase2 loader:\n{traceback.format_exc()}")

    # ── Test loaders for periodic evaluation ────────────────────
    test_loaders: dict = {}
    if args.skip_eval:
        test_loaders = None
        if is_main:
            print("\n[Eval] --skip_eval set; all evaluation disabled during training.")
    else:
        if args.curvton_test_data_path:
            genders_test = ("female", "male") if args.gender == "all" else (args.gender,)
            print(f"\nBuilding CurvTon test loaders from {args.curvton_test_data_path} ...")
            curvton_test = get_curvton_test_dataloaders(
                root_dir=args.curvton_test_data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                size=image_size,
                genders=genders_test,
            )
            # prefix curvton splits; skip 'all' since it duplicates easy+medium+hard
            for k, v in curvton_test.items():
                if k == "all":
                    continue  # 'curvton_all' is a superset of the 3 difficulty splits — skip to avoid double-counting
                test_loaders[f"curvton_{k}"] = v
            print(f"✓ CurvTon test loaders: {[k for k in test_loaders]}")
        else:
            print("\n[Eval] No --curvton_test_data_path provided; CurvTon test eval skipped.")
        if args.triplet_test_data_path:
            print(f"\nBuilding Triplet test loaders from {args.triplet_test_data_path} ...")
            triplet_test = get_triplet_test_dataloaders(
                root_dir=args.triplet_test_data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                size=image_size,
            )
            test_loaders.update(triplet_test)
            print(f"✓ Triplet test loaders: {list(triplet_test.keys())}")
        else:
            print("[Eval] No --triplet_test_data_path provided; triplet eval skipped.")
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

    # WandB (rank 0 only)
    if is_main and wandb_enabled:
        _wandb_cfg = {
            "lr": args.lr,
            "batch_size_per_gpu": args.batch_size,
            "total_batch_size": args.batch_size * world_size,
            "world_size": world_size,
            "epochs": args.epochs,
            "model": MODEL_NAME,
            "train_mode": args.train_mode,
            "dataset": dataset_label,
            "trainable_params": trainable_params,
            "curriculum": args.curriculum,
            "stage_steps": args.stage_steps,
            "data_fraction": args.data_fraction,
            "ootd": args.ootd,
            "pose_model": "MediaPipe BlazePose model_complexity=1 (Full, 33 landmarks)",
            "lr_schedule": f"cosine_anneal_{args.lr:.0e}_to_5e-05",
            "lr_eta_min": 5e-5,
            "max_steps": args.max_steps,
            "phase2_data_path": getattr(args, 'phase2_data_path', None),
            "phase2_start_step": getattr(args, 'phase2_start_step', 28801),
        }
        _init_timeout = int(os.getenv("WANDB_INIT_TIMEOUT", "600"))
        _online_retries = int(os.getenv("WANDB_ONLINE_RETRIES", "2"))
        _run = None
        _last_online_err = None
        for _attempt in range(1, max(1, _online_retries) + 1):
            try:
                _run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", WANDB_PROJECT),
                    entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
                    id=run_name,
                    resume="allow",
                    config=_wandb_cfg,
                    name=run_name,
                    settings=wandb.Settings(init_timeout=_init_timeout),
                )
                break
            except Exception as _we:
                _last_online_err = _we
                _train_log.warning(
                    "wandb.init online failed on attempt %d/%d: %s",
                    _attempt, _online_retries, _we,
                )
                if _attempt < _online_retries:
                    time.sleep(5)

        if _run is None:
            _train_log.warning(
                "wandb.init online failed after %d attempts (timeout=%ss). Retrying in offline mode.",
                _online_retries, _init_timeout,
            )
            try:
                os.environ["WANDB_MODE"] = "offline"
                _run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", WANDB_PROJECT),
                    entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
                    id=run_name,
                    resume="allow",
                    config=_wandb_cfg,
                    name=run_name,
                    settings=wandb.Settings(init_timeout=_init_timeout),
                )
                _train_log.warning("W&B started in offline mode (WANDB_MODE=offline).")
            except Exception as _we2:
                _train_log.warning(
                    "wandb.init offline also failed (last online error: %s, offline error: %s). "
                    "Continuing with W&B disabled.",
                    _last_online_err, _we2,
                )
                wandb_enabled = False

    # Optimizer (only trainable params)
    trainable_params_list = [p for p in model.unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=args.lr)
    scaler = GradScaler()

    # Cosine annealing LR
    _total_steps = (
        args.max_steps if args.max_steps > 0
        else args.epochs * len(train_loader)
    )
    _LR_ETA_MIN  = 5e-5
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(_total_steps, 1), eta_min=_LR_ETA_MIN)

    global_step = 0
    start_epoch = 0
    ema_loss    = None      # may be overwritten by checkpoint

    # ── Checkpoint Resume Logic ──────────────────────────────────
    ckpt_to_load = None
    if args.resume:
        ckpt_to_load = args.resume
        print(f"\n▶ --resume specified: {ckpt_to_load}")
    elif not args.no_resume:
        # Search both the legacy checkpoint_dir and the new experiments dir
        _exp_ckpt_dir = os.path.join("/iopsstor/scratch/cscs/dbartaula/experiments_assets", run_name, "checkpoints")
        candidates = (
            glob.glob(os.path.join(_exp_ckpt_dir, "ckpt_step_*.pt"))
            + glob.glob(os.path.join(_exp_ckpt_dir, "ckpt_final.pt"))
            + glob.glob(os.path.join(args.checkpoint_dir,
                                     f"checkpoint_vitonhd_{args.train_mode}_step_*.pt"))
        )
        if candidates:
            def _step_num(p):
                basename = os.path.basename(p)
                if basename == "ckpt_final.pt":
                    return float('inf')  # final checkpoint is always the latest
                try:
                    return int(basename.split("_step_")[1].replace(".pt", ""))
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
        _unet_raw = model.unet.module if isinstance(model.unet, DDP) else model.unet
        # Remove 'module.' prefix if present in checkpoint keys
        unet_state_dict = ckpt["unet_state_dict"]
        if any(key.startswith("module.") for key in unet_state_dict.keys()):
            unet_state_dict = {key.replace("module.", "", 1): value for key, value in unet_state_dict.items()}
        _unet_raw.load_state_dict(unet_state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if is_main:
                print(f"   ✓ LR scheduler state restored.")
        else:
            for _ in range(global_step):
                lr_scheduler.step()
            if is_main:
                print(f"   ⚠ No scheduler state — fast-forwarded {global_step} steps.")
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            if is_main:
                print(f"   ✓ GradScaler state restored.")
        if "ema_loss" in ckpt and ckpt["ema_loss"] is not None:
            ema_loss = ckpt["ema_loss"]
            if is_main:
                print(f"   ✓ EMA loss restored: {ema_loss:.6f}")
        if is_main:
            print(f"   ✓ Resumed — global_step={global_step}, start_epoch={start_epoch}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        if is_main:
            print("\n✓ No checkpoint found. Starting fresh with Xavier init on new UNet channels.")

    if is_main:
        print("\n" + "="*60)
        print(f"TRAINING: {args.epochs} epochs, {batches_per_epoch} batches/epoch per GPU")
        print(f"Mode: {args.train_mode}  |  Per-GPU batch={args.batch_size}  "
              f"Total batch={args.batch_size * world_size} ({world_size} GPUs)")
        print(f"LR: Cosine {args.lr:.0e} → {_LR_ETA_MIN:.0e} over {_total_steps} steps")
        print("="*60 + "\n")

    # ── Pre-allocate reusable tensors ───────────────────────────
    _cached_text_emb = torch.zeros(args.batch_size, 77, 768, device=device)

    # ── Running stats (window of 100 steps) ─────────────────────
    _WINDOW = 100
    loss_window      = collections.deque(maxlen=_WINDOW)
    grad_norm_window = collections.deque(maxlen=_WINDOW)
    if not isinstance(ema_loss, (float, int)):
        ema_loss = None
    EMA_DECAY        = 0.99

    # ── Phase-2 state ────────────────────────────────────────────
    _in_phase2        = False
    _phase2_announced = False
    _phase2_iter      = None
    _phase2_start_step = getattr(args, 'phase2_start_step', 28801)

    for epoch in range(start_epoch, args.epochs):
        # Update DistributedSamplers so each epoch has unique shuffling
        if world_size > 1:
            if _train_sampler is not None:
                _train_sampler.set_epoch(epoch)
            for _s in diff_samplers.values():
                if _s is not None:
                    _s.set_epoch(epoch)
            if _in_phase2 and phase2_sampler is not None:
                phase2_sampler.set_epoch(epoch)
        model.unet.train()
        epoch_losses = []

        # Curriculum: build per-difficulty iterators (reset each epoch)
        if use_curvton and args.curriculum != "none":
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
            elif use_curvton and args.curriculum != "none":
                we, wm, wh = _curriculum_weights(global_step, args.curriculum, args.stage_steps, getattr(args, 'hard_pct', None))
                _tw = we + wm + wh
                _chosen = random.choices(
                    list(CurvtonDataset.DIFFICULTIES),
                    weights=[we / _tw, wm / _tw, wh / _tw],
                )[0]
                batch = _next_curriculum_batch(_chosen)
            else:
                batch = _item

            # ── Skip incomplete/None batches ─────────────────────
            if batch is None:
                _train_log.warning(f"step={global_step}: received None batch, skipping")
                continue

            _step_ok = True
            try:
                gt         = batch['ground_truth'].to(device, non_blocking=True)
                cloth      = batch['cloth'].to(device, non_blocking=True)
                if "person" not in batch:
                    raise KeyError("Expected 'person' (initial person image) in batch.")
                person_img = batch["person"].to(device, non_blocking=True)
                # Requested conditioning behavior:
                # - masked person image -> use initial person image
                # - mask -> black tensor
                # - pose map -> use provided pose; otherwise black tensor fallback
                # No pose required: directly use the person image
                person_with_pose = person_img

                # Fused VAE encode: batch cond + target into a single forward pass
                with torch.no_grad(), autocast():
                    B = gt.shape[0]
                    if args.ootd:
                        cond_input   = cloth
                        target_input = gt
                    else:
                        cond_input   = torch.cat([person_with_pose, cloth], dim=3)
                        target_input = torch.cat([gt,         cloth], dim=3)
                    fused_input  = torch.cat([cond_input, target_input], dim=0)
                    fused_latents = model.vae.encode(fused_input).latent_dist.sample() * 0.18215
                    cond_latents   = fused_latents[:B]
                    target_latents = fused_latents[B:]

                # Diffusion
                noise     = torch.randn_like(target_latents)
                timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps,
                                          (target_latents.shape[0],), device=device).long()
                noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)

                # UNet input: channel-concat [noisy(4) ‖ cond(4)] → [B,8,64,128]
                unet_input = torch.cat([noisy_latents, cond_latents], dim=1)

                text_emb = _cached_text_emb[:B]

                # Optional DREAM-style target rectification (CatVTON-style).
                if args.use_dream:
                    if model.scheduler.config.prediction_type != "epsilon":
                        raise ValueError("DREAM is only implemented for epsilon prediction_type.")
                    with torch.no_grad(), autocast():
                        noise_pred_sg = model.unet(unet_input, timesteps, text_emb).sample
                    delta_noise = (noise - noise_pred_sg).detach() * args.dream_lambda
                    alpha_bar_t = model.scheduler.alphas_cumprod.to(
                        device=timesteps.device, dtype=noisy_latents.dtype
                    )[timesteps].view(-1, 1, 1, 1)
                    noisy_latents = noisy_latents + (1.0 - alpha_bar_t).sqrt() * delta_noise
                    noise = noise + delta_noise
                    unet_input = torch.cat([noisy_latents, cond_latents], dim=1)

                # Forward
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    if attn_score_reg is not None:
                        attn_score_reg.clear()
                    noise_pred = model.unet(unet_input, timesteps, text_emb).sample
                    loss_diff = F.mse_loss(noise_pred, noise)
                    loss_attn_reg = torch.zeros((), device=device, dtype=loss_diff.dtype)
                    loss_attn_score_reg = torch.zeros((), device=device, dtype=loss_diff.dtype)

                    if args.use_attention_regularization:
                        if model.scheduler.config.prediction_type != "epsilon":
                            raise ValueError("Attention regularization currently requires epsilon prediction_type.")

                        alpha_bar_t = model.scheduler.alphas_cumprod.to(
                            device=timesteps.device, dtype=noisy_latents.dtype
                        )[timesteps].view(-1, 1, 1, 1)
                        x0_pred = (noisy_latents - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

                        # Heuristic change map from |GT - person| in pixel space.
                        # We regularize latent reconstruction outside changed regions.
                        change_map = (gt - person_img).abs().mean(dim=1, keepdim=True)
                        change_mask = (change_map > args.attn_reg_threshold).float()
                        bg_mask = 1.0 - change_mask
                        bg_mask_latent = F.interpolate(bg_mask, size=x0_pred.shape[-2:], mode="nearest")
                        bg_denom = bg_mask_latent.sum().clamp_min(1.0)
                        loss_attn_reg = ((x0_pred - target_latents) ** 2 * bg_mask_latent).sum() / bg_denom

                    if attn_score_reg is not None:
                        loss_attn_score_reg = attn_score_reg.penalty(device).to(loss_diff.dtype)

                    loss = (
                        loss_diff
                        + args.attn_reg_lambda * loss_attn_reg
                        + args.attn_score_reg_lambda * loss_attn_score_reg
                    )

                # Backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1e9).item()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

            except Exception as _step_exc:
                _train_log.error(f"step={global_step} FAILED — skipping:\n{traceback.format_exc()}")
                _step_ok = False

            # Sync _step_ok across all ranks: if ANY rank failed, ALL skip
            # collective ops (image logging, eval) to prevent deadlocks.
            if world_size > 1:
                _ok_tensor = torch.tensor([1.0 if _step_ok else 0.0], device=device)
                dist.all_reduce(_ok_tensor, op=dist.ReduceOp.MIN)
                _step_ok = (_ok_tensor.item() > 0.5)

            # ── Update running stats (only if step succeeded) ────
            if _step_ok:
                loss_val = loss.item()
                loss_window.append(loss_val)
                grad_norm_window.append(grad_norm)

                ema_loss = loss_val if ema_loss is None else EMA_DECAY * ema_loss + (1 - EMA_DECAY) * loss_val

                loss_mean = statistics.mean(loss_window)
                loss_var  = statistics.pvariance(loss_window) if len(loss_window) > 1 else 0.0
                gn_mean   = statistics.mean(grad_norm_window)
                gn_var    = statistics.pvariance(grad_norm_window) if len(grad_norm_window) > 1 else 0.0

                current_lr = optimizer.param_groups[0]['lr']
                epoch_losses.append(loss_val)
                if is_main: _wandb_log({
                    "train/loss":              loss_val,
                    "train/loss_diff":         loss_diff.item(),
                    "train/loss_attn_reg":     loss_attn_reg.item(),
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

                # Curriculum weight logging
                if is_main and not _in_phase2 and use_curvton and args.curriculum != "none":
                    _cwe, _cwm, _cwh = _curriculum_weights(global_step, args.curriculum, args.stage_steps, getattr(args, 'hard_pct', None))
                    _wandb_log({
                        "curriculum/w_easy":   _cwe,
                        "curriculum/w_medium": _cwm,
                        "curriculum/w_hard":   _cwh,
                        "curriculum/stage":    min(global_step // max(args.stage_steps, 1),
                                                   len(_CURRIC_STAGES) - 1),
                    }, step=global_step)

            # ── Collective ops: ALL ranks must participate ────────
            # Log images with full inference (all GPUs run parallel inference)
            if _step_ok and global_step % args.image_log_interval == 0:
                log_images_distributed(
                    global_step, batch, model, cond_latents, target_latents,
                    num_inference_steps=args.num_inference_steps,
                    rank=rank, world_size=world_size,
                )

            # Save checkpoint
            if _step_ok and global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'train_mode': args.train_mode,
                    'unet_state_dict': model.unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_loss': ema_loss,
                }, ckpt_path)
                print(f"\n💾 Saved: {ckpt_path}")

            # ── Periodic test-set evaluation (all GPUs, results reduced to rank 0) ──
            if (not args.skip_eval
                    and test_loaders is not None
                    and global_step > 0
                    and global_step % args.eval_interval == 0):
                if is_main:
                    print(f"\n\U0001f4ca Running distributed test-set evaluation at step {global_step} ...")
                eval_metrics = evaluate_on_test(
                    model, test_loaders, device,
                    num_inference_steps=30,
                    eval_frac=0.01,      # 1% per sample
                    ootd=args.ootd,
                    rank=rank,
                    world_size=world_size,
                    num_eval_steps=50,
                    n_samples=10,        # 10 independent random samples → mean ± std
                )
                if is_main:
                    _wandb_log(eval_metrics, step=global_step)
                    print(f"\u2713 Eval metrics logged to W&B")

            if is_main: _wandb_log({"train/phase": 2 if _in_phase2 else 1}, step=global_step)

            if _step_ok:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            # ── Per-step console log (rank 0 only, every 10 steps) ──────────
            if _step_ok and is_main and global_step % 10 == 0:
                phase_str = "phase2" if _in_phase2 else "phase1"
                print(f"[step {global_step:>6}/{args.max_steps}] "
                      f"loss={loss_val:.4f}  ema={ema_loss:.4f}  "
                      f"mean={loss_mean:.4f}  gnorm={grad_norm:.3f}  "
                      f"lr={current_lr:.2e}  {phase_str}",
                      flush=True)

            global_step += 1

        # Epoch summary (rank 0 only)
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            avg_loss = float('nan')
        if is_main:
            _wandb_log({"train/epoch_avg_loss": avg_loss}, step=global_step)
            print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_loss:.6f}")

        # Break outer loop if max_steps reached
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save (rank 0 only)
    if is_main:
        _unet_raw = model.unet.module if isinstance(model.unet, DDP) else model.unet
        final_path = os.path.join(ckpt_dir, "ckpt_final.pt")
        torch.save({
            'step': global_step,
            'epoch': args.epochs,
            'train_mode': args.train_mode,
            'unet_state_dict': _unet_raw.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'ema_loss': ema_loss,
        }, final_path)
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE! Total steps: {global_step}")
        print(f"✓ Final checkpoint: {final_path}")
        print("="*60)

    # ── End-of-training evaluation (all ranks participate) ─────────
    if not args.skip_eval and test_loaders is not None:
        if is_main:
            print(f"\n📊 Final end-of-training evaluation (step {global_step}) ...")
        final_eval_metrics = evaluate_on_test(
            model, test_loaders, device,
            num_inference_steps=30,
            eval_frac=0.01,
            ootd=args.ootd,
            rank=rank,
            world_size=world_size,
            num_eval_steps=50,
            n_samples=10,
        )
        if is_main:
            # prefix all keys with final/ so they appear separately in W&B
            final_log = {k.replace("test/", "final/", 1): v
                         for k, v in final_eval_metrics.items()}
            _wandb_log(final_log, step=global_step)
            print("✓ Final eval metrics logged to W&B under 'final/' prefix")

    if is_main and wandb_enabled:
        wandb.finish()

    if attn_score_reg is not None:
        attn_score_reg.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Try-On Training (VITON-HD / CurvTon)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Explicit WandB run name (also used as experiments_asset subdir). "
                             "Auto-generated from args when omitted.")
    parser.add_argument("--train_mode", type=str, default="full_unet",
                        choices=["full_unet", "attention_only"],
                        help="Training mode: full_unet or attention_only")
    # ── Dataset selection ────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="vitonhd",
                        choices=["vitonhd", "curvton"],
                        help="Which dataset to train on")
    # VITON-HD
    parser.add_argument("--viton_data_path", type=str, default=None,
                        help="Path to VITON-HD root (contains train/ and test/)")
    # CurvTon
    parser.add_argument("--curvton_data_path", type=str, default=None,
                        help="Path to CurvTon dataset root (contains easy/, medium/, hard/)")
    parser.add_argument("--curvton_test_data_path", type=str, default=None,
                        help="Path to CurvTon test dataset root (same layout as train)")
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "easy_hard", "medium_hard", "all"],
                        help="CurvTon difficulty split to train on (default: all)")
    parser.add_argument("--gender", type=str, default="all",
                        choices=["female", "male", "all"],
                        help="CurvTon gender subset to train on (default: all)")
    # ── Common ───────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of DataLoader workers")
    parser.add_argument(
        "--image_size",
        type=int,
        default=CFG_IMAGE_SIZE,
        help="Training/eval resolution. Use <=0 to disable resizing and keep native image resolution.",
    )
    # ── Curriculum ────────────────────────────────────────────────
    parser.add_argument("--curriculum", type=str, default="none",
                        choices=["none", "hard", "soft", "reverse", "soft_reverse"],
                        help="Curriculum strategy: none | hard | soft | reverse | soft_reverse")
    parser.add_argument("--stage_steps", type=int, default=10000,
                        help="Steps per curriculum stage (hard / soft / reverse)")
    parser.add_argument("--stage_epochs", type=int, default=0,
                        help="Epochs per curriculum stage. If >0, overrides --stage_steps at runtime.")
    parser.add_argument("--hard_pct", type=float, default=None,
                        help="Initial % of hard samples for soft_reverse/reverse curriculum. E.g. 80 for 80%.")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use, e.g. 0.1 for 10%% (default: 1.0 = all)")
    parser.add_argument("--epochs", type=int, default=9999, help="Number of epochs (default: 9999 — use --max_steps to control termination)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--image_log_interval", type=int, default=250, help="Log images every N steps")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Run test-set evaluation every N steps (default: 100)")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Hard stop after this many gradient steps (0 = unlimited, train by epochs)")
    parser.add_argument("--phase2_data_path", type=str, default=None,
                        help="Path to triplet_dataset_train for hard-curriculum phase-2 "
                             "(vitonhd+dresscode). Only used when --curriculum hard.")
    parser.add_argument("--phase2_start_step", type=int, default=28801,
                        help="Step at which to switch to phase-2 dataset (default: 28801)")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of inference steps for full denoising")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a specific checkpoint .pt file to resume from")
    parser.add_argument("--no_resume", action="store_true", default=False,
                        help="Skip auto-checkpoint detection and always train from scratch")
    parser.add_argument("--checkpoint_dir", type=str, default=".",
                        help="Directory to scan for the latest checkpoint when --resume is not given")
    parser.add_argument("--triplet_test_data_path", type=str, default=None,
                        help="Path to triplet_dataset root for dresscode/viton-hd test eval")
    parser.add_argument("--ootd", action="store_true", default=False,
                        help="OOTDiffusion mode: cloth-only conditioning, no person image "
                             "or mask in cond (requires --dataset curvton)")
    parser.add_argument("--skip_eval", action="store_true", default=False,
                        help="Skip all evaluation during training (periodic and end-of-training). "
                             "Use evaluate.py separately to evaluate checkpoints.")
    parser.add_argument("--use_dream", action="store_true", default=False,
                        help="Enable DREAM-style target rectification during training.")
    parser.add_argument("--dream_lambda", type=float, default=10.0,
                        help="DREAM lambda strength (CatVTON paper ablation reports best around 10).")
    parser.add_argument("--use_attention_regularization", action="store_true", default=False,
                        help="Add background-preservation regularization on predicted x0 latents.")
    parser.add_argument("--attn_reg_lambda", type=float, default=1.0,
                        help="Weight for attention regularization term.")
    parser.add_argument("--attn_reg_threshold", type=float, default=0.08,
                        help="Pixel-space |GT-person| threshold to define changed regions (in [0,2]).")
    parser.add_argument("--use_attention_score_regularization", action="store_true", default=False,
                        help="Regularize attention activation magnitude as a proxy for minimizing unnecessary attention.")
    parser.add_argument("--attn_score_reg_lambda", type=float, default=1e-4,
                        help="Weight for attention-score proxy regularization.")

    args = parser.parse_args()

    # Validate that the required data path was supplied
    if args.dataset == "curvton" and not args.curvton_data_path:
        parser.error("--curvton_data_path is required when --dataset curvton")
    if args.dataset == "vitonhd" and not args.viton_data_path:
        parser.error("--viton_data_path is required when --dataset vitonhd")
    if args.ootd and args.dataset != "curvton":
        parser.error("--ootd requires --dataset curvton (needs unmasked initial person image)")

    train(args)

