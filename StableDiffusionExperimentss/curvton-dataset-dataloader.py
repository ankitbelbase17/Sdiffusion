"""
curvton-dataset-dataloader.py
==============================
Standalone CurvTon dataset & dataloader module.

Extracted from utils.py and dataloader.py without modifying either file.

Public API
----------
Dataset classes:
    CurvtonDataset          – single local leaf (one difficulty × gender)
    CombinedCurvtonDataset  – concatenated across any difficulty × gender subset

DataLoader factories:
    get_curvton_dataloaders       – shuffled train loaders per difficulty + "all"
    get_curvton_test_dataloaders  – unshuffled eval loaders per difficulty + "all"
    build_dataloaders             – unified factory used by train scripts

Helpers:
    collate_fn
    curriculum_weights
    subsample_dataset
    DataloaderBundle
    _CURRIC_STAGES

Local dataset layout expected
------------------------------
    {root_dir}/{difficulty}/{gender}/
        cloth_image/           {base}_fc_{id}_{name}.png   (female)
                               {base}_mc_{id}_{name}.png   (male)
        initial_person_image/  {person_base}.png
        tryon_image/           {base}_fc_{id}_{name}.png
                               {base}_fc_{id}_{name}.json

Where:
    difficulty ∈ {"easy", "medium", "hard"}
    gender     ∈ {"female", "male"}

Default dataset root
--------------------
    /iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate

CLI smoke-test
--------------
    python curvton-dataset-dataloader.py \\
        --curvton_data_path /iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate \\
        --batch_size 4 --num_workers 0 --epochs 1
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# IMAGE_SIZE default — overridden when imported alongside config.py
try:
    from config import IMAGE_SIZE
except ImportError:
    IMAGE_SIZE = 512
    print("[curvton-dataset-dataloader] config.py not found — using IMAGE_SIZE=512.")


# ============================================================
# LOCAL FILESYSTEM HELPERS
# ============================================================
_FC_MC_RE = re.compile(r'_(?:fc|mc)_')


def _local_load_image(path: str) -> Image.Image:
    """Load a local file and return a PIL Image (RGB)."""
    return Image.open(path).convert("RGB")


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    """
    Collate a list of sample dicts into a batched dict.
    Handles both CurvTon ('person' key) and VITON-HD ('masked_person' key).
    Drops None samples (failed S3 fetches).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError("Empty batch — all samples in this batch failed.")
    out = {
        "ground_truth": torch.stack([b["ground_truth"] for b in batch]),
        "cloth":        torch.stack([b["cloth"]        for b in batch]),
        "mask":         torch.stack([b["mask"]         for b in batch]),
    }
    if "person" in batch[0]:
        out["person"] = torch.stack([b["person"] for b in batch])
    else:
        out["masked_person"] = torch.stack([b["masked_person"] for b in batch])
    return out


# ============================================================
# CURVTON DATASET  (streams from AWS S3)
# ============================================================

class CurvtonDataset(Dataset):
    """
    Single-leaf CurvTon dataset — reads images from local filesystem.

    Local layout (one difficulty + gender leaf)::

        {root_dir}/{difficulty}/{gender}/
            cloth_image/           {base}_fc_{id}_{name}.png   (female)
                                   {base}_mc_{id}_{name}.png   (male)
            initial_person_image/  {person_base}.png
            tryon_image/           {base}_fc_{id}_{name}.png
                                   {base}_fc_{id}_{name}.json

    The person base is derived from the cloth filename by splitting on the
    first ``_fc_`` / ``_mc_`` separator, e.g.:
        ``fh_000001_e01_fc_010632_tracksuit.png``  →  person ``fh_000001_e01.png``

    Each ``__getitem__`` returns a dict:
        ground_truth  – try-on result   [3, H, W]  normalised to [-1, 1]
        cloth         – garment image   [3, H, W]  normalised to [-1, 1]
        mask          – zeros tensor    [1, H, W]
        person        – initial_image   [3, H, W]  normalised to [-1, 1]
    """

    DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
    GENDERS:      tuple[str, ...] = ("female", "male")

    def __init__(self, root_dir: str, difficulty: str = "easy",
                 gender: str = "female", size: int = 512):
        self.root_dir = root_dir
        self.size     = size
        leaf_dir = os.path.join(root_dir, difficulty, gender)

        cloth_dir  = os.path.join(leaf_dir, "cloth_image")
        person_dir = os.path.join(leaf_dir, "initial_person_image")
        tryon_dir  = os.path.join(leaf_dir, "tryon_image")

        print(f"[CurvTon-{difficulty}/{gender}] Scanning local files …")
        cloth_files = sorted(
            f for f in os.listdir(cloth_dir) if f.endswith(".png")
        )

        # Build sets of available person and tryon stems for fast O(1) lookup
        person_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(person_dir) if f.endswith(".png")
        }
        tryon_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(tryon_dir) if f.endswith(".png")
        }

        # Build (person_path, cloth_path, tryon_path) triplets
        self.triplets: list[tuple[str, str, str]] = []
        missing = 0
        for fname in cloth_files:
            stem = os.path.splitext(fname)[0]  # cloth stem
            # Derive person base: everything before first _fc_ / _mc_
            m = _FC_MC_RE.search(stem)
            if m is None:
                missing += 1
                continue
            person_base = stem[: m.start()]

            if person_base not in person_stems:
                missing += 1
                continue
            if stem not in tryon_stems:
                missing += 1
                continue

            self.triplets.append((
                os.path.join(person_dir, person_base + ".png"),
                os.path.join(cloth_dir,  fname),
                os.path.join(tryon_dir,  stem + ".png"),
            ))

        if missing:
            print(f"[CurvTon-{difficulty}/{gender}] Skipped {missing} incomplete triplets")
        print(f"[CurvTon-{difficulty}/{gender}] {len(self.triplets)} valid triplets")

        self.img_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self._zero_mask = torch.zeros(1, size, size)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        person_path, cloth_path, tryon_path = self.triplets[idx]
        try:
            person = self.img_tf(_local_load_image(person_path))
            cloth  = self.img_tf(_local_load_image(cloth_path))
            vton   = self.img_tf(_local_load_image(tryon_path))
        except Exception as e:
            print(f"[CurvTon] Load failed for triplet {idx}: {e}")
            return None

        return {
            "ground_truth": vton,            # [3,H,W] — try-on result
            "cloth":        cloth,           # [3,H,W] — garment flat-lay
            "mask":         self._zero_mask, # [1,H,W] — placeholder zeros
            "person":       person,          # [3,H,W] — initial (unmasked) person
        }


# ============================================================
# COMBINED CURVTON DATASET
# ============================================================

class CombinedCurvtonDataset(Dataset):
    """
    Concatenates :class:`CurvtonDataset` instances across any subset of
    difficulty × gender combinations from the same local root directory.

    Example::

        ds = CombinedCurvtonDataset(
            root_dir="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate",
            difficulties=("easy", "medium"),
            genders=("female",),
            size=512,
        )
    """

    def __init__(self, root_dir: str,
                 difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
                 genders: tuple[str, ...]       = ("female", "male"),
                 size: int                       = 512):
        self.datasets: list[CurvtonDataset] = []
        for diff in difficulties:
            for gender in genders:
                try:
                    ds = CurvtonDataset(root_dir, diff, gender, size)
                    self.datasets.append(ds)
                except Exception as e:
                    print(f"[CombinedCurvTon] Skipping {diff}/{gender}: {e}")

        self._cum_lengths: list[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cum_lengths.append(total)

        print(f"[CombinedCurvTon] Total triplets: {total}")

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, idx: int):
        # Binary search for the right sub-dataset
        lo, hi = 0, len(self.datasets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum_lengths[mid]:
                hi = mid
            else:
                lo = mid + 1
        offset = self._cum_lengths[lo - 1] if lo > 0 else 0
        return self.datasets[lo][idx - offset]


# ============================================================
# DATALOADER FACTORIES
# ============================================================

def _make_loader(dataset, batch_size: int, num_workers: int,
                 shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    """Standard GPU-optimised DataLoader."""
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
    )


def get_curvton_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    size: int = 512,
    genders: tuple[str, ...] = ("female", "male"),
) -> dict[str, DataLoader]:
    """
    Shuffled train dataloaders per difficulty + combined ``"all"``.

    Parameters
    ----------
    root_dir    : Absolute path to the local dataset root.
    batch_size  : Samples per batch.
    num_workers : DataLoader worker processes.
    size        : Image resize target (square).
    genders     : Subset of ``("female", "male")`` to include.

    Returns
    -------
    dict with keys ``"easy"``, ``"medium"``, ``"hard"``, ``"all"``.
    """
    loaders: dict[str, DataLoader] = {}
    for diff in CurvtonDataset.DIFFICULTIES:
        ds = CombinedCurvtonDataset(root_dir, difficulties=(diff,),
                                    genders=genders, size=size)
        loaders[diff] = _make_loader(ds, batch_size, num_workers,
                                     shuffle=True, drop_last=True)
        print(f"[Train DataLoader] {diff}: {len(loaders[diff])} batches/epoch")

    all_ds = CombinedCurvtonDataset(root_dir,
                                    difficulties=CurvtonDataset.DIFFICULTIES,
                                    genders=genders, size=size)
    loaders["all"] = _make_loader(all_ds, batch_size, num_workers,
                                  shuffle=True, drop_last=True)
    print(f"[Train DataLoader] all:  {len(loaders['all'])} batches/epoch")
    return loaders


def get_curvton_test_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    size: int = 512,
    genders: tuple[str, ...] = ("female", "male"),
) -> dict[str, DataLoader]:
    """
    Unshuffled evaluation dataloaders per difficulty + combined ``"all"``.

    ``shuffle=False, drop_last=False`` — every sample is evaluated exactly once.

    Parameters
    ----------
    root_dir    : Absolute path to the local test dataset root.
    batch_size  : Samples per batch.
    num_workers : DataLoader worker processes.
    size        : Image resize target (square).
    genders     : Subset of ``("female", "male")`` to include.

    Returns
    -------
    dict with keys ``"easy"``, ``"medium"``, ``"hard"``, ``"all"``.
    """
    loaders: dict[str, DataLoader] = {}
    for diff in CurvtonDataset.DIFFICULTIES:
        ds = CombinedCurvtonDataset(root_dir, difficulties=(diff,),
                                    genders=genders, size=size)
        loaders[diff] = _make_loader(ds, batch_size, num_workers,
                                     shuffle=False, drop_last=False)
        print(f"[Test DataLoader] {diff}: {len(loaders[diff])} batches")

    all_ds = CombinedCurvtonDataset(root_dir,
                                    difficulties=CurvtonDataset.DIFFICULTIES,
                                    genders=genders, size=size)
    loaders["all"] = DataLoader(all_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=False,
                                collate_fn=collate_fn)
    print(f"[Test DataLoader] all:  {len(loaders['all'])} batches")
    return loaders


# ============================================================
# CURRICULUM TRAINING HELPERS
# ============================================================

# Stage schedules: (w_easy, w_medium, w_hard) — normalised at runtime.
# Default boundary (--stage_steps 7000): transitions at 7k, 14k, 21k steps.
_CURRIC_STAGES = [
    (1.0, 0.0, 0.0),    # stage 0 — easy only
    (0.3, 0.7, 0.0),    # stage 1 — mostly medium
    (0.2, 0.3, 0.5),    # stage 2 — hard-skewed
    (0.2, 0.3, 0.4),    # stage 3 — 20% easy | 30% medium | 40% hard
]
_REVERSE_STAGES = [
    (0.0, 0.0, 1.0),    # stage 0 — hard only
    (0.0, 0.3, 0.7),    # stage 1 — mostly hard
    (0.2, 0.3, 0.5),    # stage 2 — same end-state as forward curriculum
    (0.34, 0.33, 0.33), # stage 3 — fully balanced
]


def curriculum_weights(
    step: int, curriculum: str, stage_steps: int
) -> tuple[float, float, float]:
    """
    Return ``(w_easy, w_medium, w_hard)`` for the current training step.

    Parameters
    ----------
    step        : Current global training step.
    curriculum  : One of ``"none"``, ``"hard"``, ``"soft"``, ``"reverse"``.
    stage_steps : Number of steps per curriculum stage.

    Returns
    -------
    Three floats (unnormalised weights for easy, medium, hard).
    """
    if curriculum == "none":
        return (1.0, 1.0, 1.0)
    stages = _REVERSE_STAGES if curriculum == "reverse" else _CURRIC_STAGES
    frac = step / max(stage_steps, 1)
    lo   = min(int(frac), len(stages) - 1)
    hi   = min(lo + 1,   len(stages) - 1)
    if curriculum in ("hard", "reverse"):
        return stages[lo]
    # soft: linear blend
    t  = frac - int(frac)
    we = stages[lo][0] * (1.0 - t) + stages[hi][0] * t
    wm = stages[lo][1] * (1.0 - t) + stages[hi][1] * t
    wh = stages[lo][2] * (1.0 - t) + stages[hi][2] * t
    return (we, wm, wh)


def subsample_dataset(dataset, fraction: float, seed: int = 42):
    """Return a random :class:`~torch.utils.data.Subset` of ``fraction`` of ``dataset``."""
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    print(f"  ↳ Sub-sampled {k}/{n} samples ({fraction * 100:.0f}%)")
    return Subset(dataset, indices)


# ============================================================
# HIGH-LEVEL BUNDLE  (mirrors dataloader.DataloaderBundle)
# ============================================================

@dataclass
class DataloaderBundle:
    """
    Container returned by :func:`build_dataloaders`.

    Attributes
    ----------
    train_loader      : Shuffled DataLoader over all selected difficulties.
    diff_loaders      : Per-difficulty DataLoaders ``{"easy", "medium", "hard"}``.
    test_loaders      : Per-difficulty + ``"all"`` test DataLoaders, or ``None``.
    batches_per_epoch : ``len(train_loader)``.
    dataset_label     : Human-readable tag for W&B / logging.
    """
    train_loader:      DataLoader
    diff_loaders:      Dict[str, DataLoader]
    test_loaders:      Optional[Dict[str, DataLoader]]
    batches_per_epoch: int
    dataset_label:     str


def _resolve_genders(args: argparse.Namespace) -> tuple[str, ...]:
    gender = getattr(args, "gender", "all")
    return ("female", "male") if gender == "all" else (gender,)


def build_dataloaders(args: argparse.Namespace) -> DataloaderBundle:
    """
    Unified dataloader factory — builds train, per-difficulty, and test loaders.

    Expected ``args`` attributes
    ----------------------------
    curvton_data_path      : str  – Local dataset root path.
    curvton_test_data_path : str  – Local test dataset root (optional)
    difficulty             : str  – ``"easy" | "medium" | "hard" | "all"``
    gender                 : str  – ``"female" | "male" | "all"``
    batch_size             : int
    num_workers            : int
    data_fraction          : float – fraction of dataset to use (1.0 = all)
    curriculum             : str  – ``"none" | "hard" | "soft" | "reverse"``

    Returns
    -------
    :class:`DataloaderBundle`
    """
    genders  = _resolve_genders(args)
    root_dir = args.curvton_data_path
    frac     = getattr(args, "data_fraction", 1.0)
    diff     = getattr(args, "difficulty",    "all")

    # ── Per-difficulty train loaders (always built for curriculum sampling) ──
    diff_loaders: Dict[str, DataLoader] = {}
    for _diff in CurvtonDataset.DIFFICULTIES:
        _ds = CombinedCurvtonDataset(root_dir=root_dir, difficulties=(_diff,),
                                     genders=genders, size=IMAGE_SIZE)
        _ds = subsample_dataset(_ds, frac)
        diff_loaders[_diff] = _make_loader(_ds, args.batch_size, args.num_workers)
        print(f"✓ CurvTon [{_diff}] DataLoader: {len(diff_loaders[_diff])} batches/epoch")

    # ── Combined train loader (selected difficulty or all) ────────────────
    difficulties = CurvtonDataset.DIFFICULTIES if diff == "all" else (diff,)
    train_ds = CombinedCurvtonDataset(root_dir=root_dir, difficulties=difficulties,
                                      genders=genders, size=IMAGE_SIZE)
    train_ds = subsample_dataset(train_ds, frac)
    train_loader      = _make_loader(train_ds, args.batch_size, args.num_workers)
    batches_per_epoch = len(train_loader)

    curriculum    = getattr(args, "curriculum", "none")
    frac_tag      = f"_frac{frac}" if frac < 1.0 else ""
    gender_str    = getattr(args, "gender", "all")
    dataset_label = f"CurvTon-{diff}-{gender_str}-{curriculum}{frac_tag}"
    print(f"✓ Train DataLoader ({dataset_label}): {batches_per_epoch} batches/epoch")

    # ── Optional test loaders ────────────────────────────────────────────
    test_loaders = None
    test_path    = getattr(args, "curvton_test_data_path", None)
    if test_path:
        print(f"\nBuilding CurvTon test loaders from {test_path} …")
        test_loaders = get_curvton_test_dataloaders(
            root_dir=test_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
            genders=genders,
        )
        print(f"✓ Test loaders built: {list(test_loaders.keys())}")
    else:
        print("\n[Eval] No --curvton_test_data_path provided; test evaluation skipped.")

    return DataloaderBundle(
        train_loader=train_loader,
        diff_loaders=diff_loaders,
        test_loaders=test_loaders,
        batches_per_epoch=batches_per_epoch,
        dataset_label=dataset_label,
    )


# ============================================================
# CLI SMOKE-TEST  (python curvton-dataset-dataloader.py --help)
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test: build CurvTon dataloaders and iterate one epoch."
    )
    p.add_argument("--curvton_data_path",      type=str, required=True, metavar="LOCAL_PATH",
                   help='Absolute path to dataset root, e.g. "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"')
    p.add_argument("--curvton_test_data_path", type=str, default=None)
    p.add_argument("--difficulty",   choices=["easy", "medium", "hard", "all"], default="all")
    p.add_argument("--gender",       choices=["female", "male", "all"],         default="all")
    p.add_argument("--data_fraction",type=float, default=1.0)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--curriculum",   choices=["none", "hard", "soft", "reverse"], default="none")
    p.add_argument("--stage_steps",  type=int,   default=7000)
    p.add_argument("--epochs",       type=int,   default=1,
                   help="Number of epochs to iterate for throughput measurement.")
    return p.parse_args()


if __name__ == "__main__":
    import time

    args = _parse_args()
    print("=" * 60)
    print("CurvTon Dataset Dataloader — smoke-test")
    print("=" * 60)

    bundle = build_dataloaders(args)

    print(f"\nBundle summary:")
    print(f"  dataset_label     : {bundle.dataset_label}")
    print(f"  batches_per_epoch : {bundle.batches_per_epoch}")
    print(f"  diff_loaders      : {list(bundle.diff_loaders.keys())}")
    print(f"  test_loaders      : "
          f"{list(bundle.test_loaders.keys()) if bundle.test_loaders else None}")

    print(f"\nIterating {args.epochs} epoch(s) of train_loader …")
    t0 = time.perf_counter()
    total_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(bundle.train_loader):
            if epoch == 0 and step == 0:
                print("\nFirst batch tensor shapes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k:20s} : {list(v.shape)}  dtype={v.dtype}")
            total_step += 1

    elapsed      = time.perf_counter() - t0
    total_samples = total_step * args.batch_size
    print(f"\n✓ {total_step} steps | {total_samples} samples | "
          f"{elapsed:.1f}s | {total_samples / elapsed:.1f} samples/s")

    if args.curriculum != "none":
        print(f"\nCurriculum weight progression ({args.curriculum}):")
        for step in [0, args.stage_steps, args.stage_steps * 2, args.stage_steps * 3]:
            we, wm, wh = curriculum_weights(step, args.curriculum, args.stage_steps)
            total = we + wm + wh
            print(f"  step {step:6d}  → easy={we/total:.2f}  medium={wm/total:.2f}  hard={wh/total:.2f}")
