"""
dataloader.py — Unified dataloader factory for all virtual try-on training scripts.

Public API
----------
build_dataloaders(args)  →  DataloaderBundle
    Single call used by train.py, train_tryongan.py, and train_DiT.py.
    Returns a DataloaderBundle (named-tuple) with:
        train_loader      – shuffled main DataLoader (all selected difficulties)
        diff_loaders      – dict{"easy"|"medium"|"hard": DataLoader}
                            (one per difficulty; required for curriculum sampling)
        test_loaders      – dict{"easy"|"medium"|"hard"|"all": DataLoader} or None
        batches_per_epoch – int  (len of train_loader)
        dataset_label     – str  (human-readable tag for W&B config)

Dataset classes (re-exported for direct import)
-------------------------------------------------
    VitonHDDataset
    CurvtonDataset
    CombinedCurvtonDataset
    collate_fn

Curriculum helpers (re-exported)
---------------------------------
    curriculum_weights
    subsample_dataset
    _CURRIC_STAGES

CLI smoke-test
--------------
    python dataloader.py \\
        --curvton_data_path /iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate \\
        --batch_size 4 --num_workers 0 --epochs 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

try:
    from config import IMAGE_SIZE
except Exception:
    IMAGE_SIZE = 512
from utils import (
    # Dataset classes
    VitonHDDataset,
    CurvtonDataset,
    CombinedCurvtonDataset,
    collate_fn,
    # Dataloader factories
    get_curvton_test_dataloaders,
    # Utilities
    curriculum_weights,
    subsample_dataset,
    _CURRIC_STAGES,
)

__all__ = [
    # Main factory
    "build_dataloaders",
    "DataloaderBundle",
    # Dataset classes
    "VitonHDDataset",
    "CurvtonDataset",
    "CombinedCurvtonDataset",
    "collate_fn",
    # Helpers
    "curriculum_weights",
    "subsample_dataset",
    "_CURRIC_STAGES",
]


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class DataloaderBundle:
    """
    Everything a training script needs from dataloader setup.

    Attributes
    ----------
    train_loader      : shuffled DataLoader over the selected combined dataset.
    diff_loaders      : per-difficulty DataLoaders {"easy", "medium", "hard"}.
                        Always present — used for curriculum batch selection.
    test_loaders      : per-difficulty + "all" test DataLoaders, or None.
    batches_per_epoch : len(train_loader).
    dataset_label     : human-readable string for W&B logging.
    """
    train_loader:      DataLoader
    diff_loaders:      Dict[str, DataLoader]
    test_loaders:      Optional[Dict[str, DataLoader]]
    batches_per_epoch: int
    dataset_label:     str


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with the project-standard GPU-optimised settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        worker_init_fn=None,
    )


# ============================================================
# MAIN PUBLIC FACTORY
# ============================================================

def build_dataloaders(args: argparse.Namespace) -> DataloaderBundle:
    """
    Build all DataLoaders required for a training run.

    Expected ``args`` attributes (all training scripts share these)
    ---------------------------------------------------------------
    curvton_data_path      str | None   – Local path to dataset root, e.g.
                                          "/iopsstor/scratch/cscs/dbartaula/human_gen/
                                          dataset_v3_backup_1/dataset_ultimate"
                                          If None, falls back to VITON-HD.
    viton_data_path        str | None   – Local VITON-HD root dir.
    curvton_test_data_path str | None   – Local path for test split (optional).
    difficulty             str          – "easy" | "medium" | "hard" | "all"
    gender                 str          – "female" | "male" | "all"
    batch_size             int
    num_workers            int
    data_fraction          float        – fraction of dataset to use (1.0 = all)
    curriculum             str          – "none" | "hard" | "soft" | "reverse"

    Returns
    -------
    DataloaderBundle
    """
    use_curvton = bool(getattr(args, "curvton_data_path", None))

    if use_curvton:
        bundle = _build_curvton(args)
    else:
        bundle = _build_vitonhd(args)

    # ── Test loaders (optional) ──────────────────────────────────
    test_loaders = None
    test_path = getattr(args, "curvton_test_data_path", None)
    if test_path:
        genders_test = _resolve_genders(args)
        print(f"\nBuilding CurvTon test loaders from {test_path} ...")
        test_loaders = get_curvton_test_dataloaders(
            root_dir=test_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
            genders=genders_test,
        )
        print(f"✓ Test loaders built: {list(test_loaders.keys())}")
    else:
        print("\n[Eval] No --curvton_test_data_path provided; test evaluation will be skipped.")

    return DataloaderBundle(
        train_loader=bundle["train_loader"],
        diff_loaders=bundle["diff_loaders"],
        test_loaders=test_loaders,
        batches_per_epoch=bundle["batches_per_epoch"],
        dataset_label=bundle["dataset_label"],
    )


# ============================================================
# CURVTON BUILDER
# ============================================================

def _resolve_genders(args: argparse.Namespace) -> tuple[str, ...]:
    gender = getattr(args, "gender", "all")
    return ("female", "male") if gender == "all" else (gender,)


def _build_curvton(args: argparse.Namespace) -> dict:
    genders  = _resolve_genders(args)
    root_dir = args.curvton_data_path
    frac     = getattr(args, "data_fraction", 1.0)
    diff     = getattr(args, "difficulty", "all")

    # ── Per-difficulty loaders (always built — needed for curriculum) ──
    diff_loaders: Dict[str, DataLoader] = {}
    for _diff in CurvtonDataset.DIFFICULTIES:
        _ds = CombinedCurvtonDataset(
            root_dir=root_dir,
            difficulties=(_diff,),
            genders=genders,
            size=IMAGE_SIZE,
        )
        _ds = subsample_dataset(_ds, frac)
        diff_loaders[_diff] = _make_loader(_ds, args.batch_size, args.num_workers)
        print(f"✓ CurvTon [{_diff}] DataLoader: {len(diff_loaders[_diff])} batches/epoch")

    # ── Combined "train" loader (selected difficulty or all) ──────
    difficulties = CurvtonDataset.DIFFICULTIES if diff == "all" else (diff,)
    train_ds = CombinedCurvtonDataset(
        root_dir=root_dir,
        difficulties=difficulties,
        genders=genders,
        size=IMAGE_SIZE,
    )
    train_ds = subsample_dataset(train_ds, frac)

    train_loader = _make_loader(train_ds, args.batch_size, args.num_workers)
    batches_per_epoch = len(train_loader)

    curriculum  = getattr(args, "curriculum", "none")
    frac_tag    = f"_frac{frac}" if frac < 1.0 else ""
    gender_str  = getattr(args, "gender", "all")
    dataset_label = f"CurvTon-{diff}-{gender_str}-{curriculum}{frac_tag}"

    print(f"✓ Train DataLoader ({dataset_label}): {batches_per_epoch} batches/epoch")

    return dict(
        train_loader=train_loader,
        diff_loaders=diff_loaders,
        batches_per_epoch=batches_per_epoch,
        dataset_label=dataset_label,
    )


# ============================================================
# VITON-HD BUILDER
# ============================================================

def _build_vitonhd(args: argparse.Namespace) -> dict:
    viton_path = getattr(args, "viton_data_path", None)
    if not viton_path:
        raise ValueError(
            "Neither --curvton_data_path nor --viton_data_path was provided."
        )

    frac = getattr(args, "data_fraction", 1.0)
    train_ds = VitonHDDataset(root_dir=viton_path, split="train", size=IMAGE_SIZE)
    train_ds = subsample_dataset(train_ds, frac)

    train_loader = _make_loader(train_ds, args.batch_size, args.num_workers)
    batches_per_epoch = len(train_loader)

    frac_tag      = f"_frac{frac}" if frac < 1.0 else ""
    dataset_label = f"VITON-HD{frac_tag}"

    # VITON-HD has no per-difficulty splits; create a single stub
    diff_loaders = {d: train_loader for d in CurvtonDataset.DIFFICULTIES}

    print(f"✓ VITON-HD DataLoader ({dataset_label}): {batches_per_epoch} batches/epoch")
    print("  ⚠  VITON-HD has no difficulty splits — diff_loaders all point to train_loader.")

    return dict(
        train_loader=train_loader,
        diff_loaders=diff_loaders,
        batches_per_epoch=batches_per_epoch,
        dataset_label=dataset_label,
    )


# ============================================================
# STANDALONE SMOKE-TEST  (python dataloader.py --help)
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test: build dataloaders and iterate one epoch."
    )
    # Dataset source
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--curvton_data_path",  type=str, metavar="LOCAL_PATH",
                   help='Local dataset root, e.g. "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"')
    g.add_argument("--viton_data_path",    type=str, metavar="LOCAL_DIR",
                   help="VITON-HD local root directory")
    # Dataset filters
    p.add_argument("--curvton_test_data_path", type=str, default=None)
    p.add_argument("--difficulty",   choices=["easy", "medium", "hard", "all"], default="all")
    p.add_argument("--gender",       choices=["female", "male", "all"],         default="all")
    p.add_argument("--data_fraction",type=float, default=1.0)
    # Loader settings
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--curriculum",   choices=["none", "hard", "soft", "reverse"], default="none")
    p.add_argument("--stage_steps",  type=int, default=4000)
    p.add_argument("--epochs",       type=int, default=1,
                   help="Number of epochs to iterate for throughput measurement")
    return p.parse_args()


if __name__ == "__main__":
    import time

    args = _parse_args()
    print("=" * 60)
    print("Dataloader smoke-test")
    print("=" * 60)

    bundle = build_dataloaders(args)

    print(f"\nBundle summary:")
    print(f"  dataset_label     : {bundle.dataset_label}")
    print(f"  batches_per_epoch : {bundle.batches_per_epoch}")
    print(f"  diff_loaders      : {list(bundle.diff_loaders.keys())}")
    print(f"  test_loaders      : {list(bundle.test_loaders.keys()) if bundle.test_loaders else None}")

    # Iterate one epoch and print first batch shapes
    print(f"\nIterating {args.epochs} epoch(s) of train_loader ...")
    t0         = time.perf_counter()
    total_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(bundle.train_loader):
            if epoch == 0 and step == 0:
                print("\nFirst batch tensor shapes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k:20s} : {list(v.shape)}  dtype={v.dtype}")
            total_step += 1

    elapsed = time.perf_counter() - t0
    total_samples = total_step * args.batch_size
    print(f"\n✓ {total_step} steps | {total_samples} samples | "
          f"{elapsed:.1f}s | {total_samples / elapsed:.1f} samples/s")

    # Quick curriculum weight preview
    if args.curriculum != "none":
        print(f"\nCurriculum weight progression ({args.curriculum}):")
        for step in [0, args.stage_steps, args.stage_steps * 2, args.stage_steps * 3]:
            we, wm, wh = curriculum_weights(step, args.curriculum, args.stage_steps)
            total = we + wm + wh
            print(f"  step {step:6d}  → easy={we/total:.2f}  medium={wm/total:.2f}  hard={wh/total:.2f}")
