"""
utils.py — Datasets, dataloaders, evaluation, inference helpers, and curriculum logic.
"""

import math
import os
import random
import re
import time
import traceback
import logging
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from torch.amp import autocast as _amp_autocast
    def _eval_autocast(): return _amp_autocast('cuda')
except ImportError:
    from torch.cuda.amp import autocast as _eval_autocast_cls
    def _eval_autocast(): return _eval_autocast_cls()
from torchvision import transforms
import wandb
try:
    from transformers import AutoImageProcessor, DetrForObjectDetection
    _DETR_AVAILABLE = True
except Exception:
    _DETR_AVAILABLE = False

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
_log = logging.getLogger("utils")

try:
    import lpips as lpips_lib
except ImportError:
    raise ImportError("Install lpips: pip install lpips")

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        raise AttributeError("mediapipe.solutions.pose not found (API removed in >= 0.10.21)")
    _mp_pose = mp.solutions.pose
    _POSE_AVAILABLE = True
except ImportError:
    _POSE_AVAILABLE = False
    print("[Warning] mediapipe not installed — pose keypoint error will be skipped. "
          "Install with: pip install 'mediapipe<=0.10.20'")
except AttributeError as _mp_err:
    _POSE_AVAILABLE = False
    print(f"[Warning] {_mp_err} — pose keypoint error will be skipped. "
          "Downgrade with: pip install 'mediapipe<=0.10.20'")

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
    from config import IMAGE_SIZE
except Exception:
    IMAGE_SIZE = 512


# ============================================================
# VITON-HD DATASET
# ============================================================
class VitonHDDataset(Dataset):
    """
    VITON-HD Dataset for Virtual Try-On

    Returns:
        - ground_truth: Original person image
        - cloth: Cloth image
        - mask: Binary mask of cloth area
        - masked_person: Person with cloth area grayed out
    """

    def __init__(self, root_dir, split='train', size=512):
        import os
        self.root_dir = root_dir
        self.split = split
        self.size = size

        # Define paths
        self.image_dir = os.path.join(root_dir, split, 'image')
        self.cloth_dir = os.path.join(root_dir, split, 'cloth')
        self.mask_dir = os.path.join(root_dir, split, 'gt_cloth_warped_mask')

        # Get all image filenames (sorted for alignment)
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])

        print(f"[VitonHD-{split}] Loaded {len(self.image_files)} samples")

        # Image transforms
        if size and size > 0:
            self.image_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        # Mask transforms (nearest neighbor to preserve binary values)
        if size and size > 0:
            self.mask_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 20 attempts: first tries the requested idx, then random fallbacks.
        # With 20 random draws from a large dataset the probability of all
        # failing is essentially zero in practice.
        _MAX_RETRIES = 20
        for _attempt in range(_MAX_RETRIES):
            _try_idx = idx if _attempt == 0 else random.randint(0, len(self) - 1)
            try:
                return self._load_single(_try_idx)
            except Exception:
                _log.warning("VitonHDDataset __getitem__ attempt %d/%d idx=%d failed:\n%s",
                             _attempt + 1, _MAX_RETRIES, _try_idx, traceback.format_exc())
        _log.error("VitonHDDataset __getitem__: all %d retries exhausted for original idx=%d — "
                   "returning None; collate_fn will replace this slot", _MAX_RETRIES, idx)
        return None

    def _load_single(self, idx):
        img_name = self.image_files[idx]

        # Load person image
        img_path = os.path.join(self.image_dir, img_name)
        person_img = Image.open(img_path).convert('RGB')
        person_tensor = self.image_transform(person_img)

        # Load cloth image
        cloth_path = os.path.join(self.cloth_dir, img_name)
        cloth_img = Image.open(cloth_path).convert('RGB')
        cloth_tensor = self.image_transform(cloth_img)

        # Load mask - try different naming patterns
        mask_img = None
        base_name = os.path.splitext(img_name)[0]

        # Try different mask naming patterns
        possible_mask_names = [
            f"{base_name}.png",           # 11386_00.png
            f"{base_name}.jpg",           # 11386_00.jpg
            img_name,                      # Same as image
            f"{base_name}_mask.png",      # 11386_00_mask.png
        ]

        for mask_name in possible_mask_names:
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                break

        # If still not found, create empty mask
        if mask_img is None:
            if self.size and self.size > 0:
                mask_img = Image.new('L', (self.size, self.size), 0)
            else:
                mask_img = Image.new('L', person_img.size, 0)
            # Only print warning once per file
            if not hasattr(self, '_warned_masks'):
                self._warned_masks = set()
            if img_name not in self._warned_masks:
                print(f"Warning: Mask not found for {img_name}")
                self._warned_masks.add(img_name)

        mask_tensor = self.mask_transform(mask_img)

        # Create masked person (gray out cloth area)
        masked_person = self._create_masked_person(person_tensor, mask_tensor)

        return {
            'ground_truth':  person_tensor,    # [3, H, W] normalized to [-1, 1]  (= tryon image)
            'person':        person_tensor,    # [3, H, W] initial_person_image (same as GT for VitonHD)
            'cloth':         cloth_tensor,     # [3, H, W] normalized to [-1, 1]
            'mask':          mask_tensor,      # [1, H, W] in range [0, 1]
            'masked_person': masked_person,    # [3, H, W] normalized to [-1, 1]
        }

    def _create_masked_person(self, person_tensor, mask_tensor):
        """
        Mask out cloth area by setting it to gray

        Args:
            person_tensor: [3, H, W] normalized to [-1, 1]
            mask_tensor: [1, H, W] in range [0, 1]

        Returns:
            masked_person: [3, H, W] with cloth area grayed out
        """
        mask_3ch = mask_tensor.expand(3, -1, -1)
        gray_value = 0.0  # Gray in normalized [-1, 1] space
        masked_person = person_tensor * (1 - mask_3ch) + gray_value * mask_3ch
        return masked_person


def collate_fn(batch):
    """Filter None/incomplete samples, pad back to original batch size, then collate.

    If any sample is None/invalid, valid samples are repeated to refill the
    batch so the batch tensor always has exactly the originally requested size.
    Returns None only when *every* sample in the batch is invalid — the
    training loop should then skip that step.
    """
    requested = len(batch)
    good = [b for b in batch if b is not None]
    if not good:
        _log.warning("collate_fn: entire batch of %d samples was None/invalid — skipping",
                     requested)
        return None
    if len(good) < requested:
        n_bad = requested - len(good)
        _log.warning("collate_fn: %d/%d bad samples — padding batch back to %d by repeating valid samples",
                     n_bad, requested, requested)
        # Repeat valid samples (cycling) to restore original batch size
        good = good + random.choices(good, k=n_bad)
    try:
        out = {
            'ground_truth': torch.stack([b['ground_truth'] for b in good]),
            'cloth':        torch.stack([b['cloth']        for b in good]),
            'mask':         torch.stack([b['mask']         for b in good]),
        }
        if 'person' in good[0]:
            out['person'] = torch.stack([b['person'] for b in good])
        if 'masked_person' in good[0]:
            out['masked_person'] = torch.stack([b['masked_person'] for b in good])
        return out
    except Exception:
        _log.error("collate_fn: stacking failed:\n%s", traceback.format_exc())
        return None


# ============================================================
# CURVTON DATASET  (reads from local filesystem)
# ============================================================
_FC_MC_RE = re.compile(r'_(?:fc|mc)_')


def _local_load_image(path: str) -> Image.Image:
    """Load a local file and return a PIL Image (RGB)."""
    return Image.open(path).convert("RGB")


_DETR_CACHE = {}


def _person_bbox_square_from_heuristic(person_img: Image.Image, margin: float = 0.15):
    arr = np.asarray(person_img.convert("RGB"))
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)
    fg = (gray > 8) & (gray < 247)
    ys, xs = np.where(fg)
    if ys.size < 16 or xs.size < 16:
        return (0, 0, w, h)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    side = int(round(max(bw, bh) * (1.0 + margin)))
    side = max(16, min(side, max(w, h)))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))
    right = min(w, left + side)
    bottom = min(h, top + side)
    return (left, top, right, bottom)


def _get_detr_detector(device: str = "cpu"):
    key = str(device)
    if key in _DETR_CACHE:
        return _DETR_CACHE[key]
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.to(device).eval()
    _DETR_CACHE[key] = (processor, model)
    return processor, model


def _person_bbox_square_from_image(person_img: Image.Image, margin: float = 0.15):
    """Estimate person square crop using DETR person detection with fallback."""
    arr = np.asarray(person_img.convert("RGB"))
    h, w = arr.shape[:2]
    x0 = y0 = x1 = y1 = None

    if _DETR_AVAILABLE:
        try:
            processor, model = _get_detr_detector("cpu")
            inputs = processor(images=person_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([[h, w]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.6)[0]

            person_boxes = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                lbl = int(label.item())
                label_name = model.config.id2label.get(lbl, "").lower()
                if label_name == "person":
                    person_boxes.append((float(score.item()), box.tolist()))
            if person_boxes:
                person_boxes.sort(key=lambda t: t[0], reverse=True)
                _, best = person_boxes[0]
                x0, y0, x1, y1 = [int(round(v)) for v in best]
                x0 = max(0, min(x0, w - 1))
                y0 = max(0, min(y0, h - 1))
                x1 = max(x0 + 1, min(x1, w))
                y1 = max(y0 + 1, min(y1, h))
        except Exception:
            x0 = y0 = x1 = y1 = None

    if x0 is None:
        return _person_bbox_square_from_heuristic(person_img, margin=margin)

    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    side = int(round(max(bw, bh) * (1.0 + margin)))
    side = max(16, min(side, max(w, h)))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))

    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))
    right = min(w, left + side)
    bottom = min(h, top + side)
    return (left, top, right, bottom)


def _eval_triplet_person_aware_tensor(
    person_img: Image.Image,
    cloth_img: Image.Image,
    tryon_img: Image.Image,
    out_size: int,
    pre_resize_size: int = 768,
):
    """Triplet eval preprocessing with preserved geometry (no crop, no resize)."""
    _ = out_size
    _ = pre_resize_size
    final_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return final_tf(person_img), final_tf(cloth_img), final_tf(tryon_img)


class CurvtonDataset(Dataset):
    """
    CurvTon Dataset — reads images from local filesystem.

    Local layout (one difficulty + gender leaf)::

        {root_dir}/{difficulty}/{gender}/
            cloth_image/           {base}_fc_{id}_{name}.png   (female)
                                   {base}_mc_{id}_{name}.png   (male)
            initial_person_image/  {person_base}.png
            tryon_image/           {base}_fc_{id}_{name}.png
                                   {base}_fc_{id}_{name}.json

    The person base is derived from the cloth filename by splitting on
    the first ``_fc_`` / ``_mc_`` separator.

    Returns dict with keys:
        ground_truth  – try-on result   [3,H,W]  normalised [-1,1]
        cloth         – garment image   [3,H,W]
        mask          – zeros           [1,H,W]
        person        – initial_image   [3,H,W]  (NOT masked)
    """

    DIFFICULTIES = ("easy", "medium", "hard")
    GENDERS      = ("female", "male")

    def __init__(self, root_dir: str, difficulty="easy", gender="female", size=512, eval_mode=False):
        self.root_dir = root_dir
        self.size     = size
        self.eval_mode = eval_mode
        leaf_dir = os.path.join(root_dir, difficulty, gender)

        cloth_dir  = os.path.join(leaf_dir, "cloth_image")
        person_dir = os.path.join(leaf_dir, "initial_person_image")
        tryon_dir  = os.path.join(leaf_dir, "tryon_image")

        print(f"[CurvTon-{difficulty}/{gender}] Scanning local files …")
        cloth_files = sorted(
            f for f in os.listdir(cloth_dir) if f.endswith(".png")
        )

        # Build a set of available person filenames (stems) for fast lookup
        person_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(person_dir) if f.endswith(".png")
        }
        # Build a set of available tryon stems (PNG only)
        tryon_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(tryon_dir) if f.endswith(".png")
        }

        # Build (person_path, cloth_path, tryon_path) triplets
        self.triplets: list[tuple[str, str, str]] = []
        missing = 0
        for fname in cloth_files:
            stem       = os.path.splitext(fname)[0]      # cloth stem
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

        # Image transform
        if size and size > 0:
            self.img_tf = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        _MAX_RETRIES = 20
        for _attempt in range(_MAX_RETRIES):
            _try_idx = idx if _attempt == 0 else random.randint(0, len(self) - 1)
            try:
                person_path, cloth_path, tryon_path = self.triplets[_try_idx]
                person_img = _local_load_image(person_path)
                cloth_img = _local_load_image(cloth_path)
                tryon_img = _local_load_image(tryon_path)
                if self.eval_mode:
                    person, cloth, vton = _eval_triplet_person_aware_tensor(
                        person_img, cloth_img, tryon_img, out_size=self.size, pre_resize_size=768
                    )
                else:
                    person = self.img_tf(person_img)
                    cloth = self.img_tf(cloth_img)
                    vton = self.img_tf(tryon_img)
                return {
                    "ground_truth": vton,
                    "cloth":        cloth,
                    "mask":         torch.zeros(1, person.shape[1], person.shape[2]),
                    "person":       person,
                }
            except Exception:
                _log.warning("CurvtonDataset __getitem__ attempt %d/%d idx=%d failed:\n%s",
                             _attempt + 1, _MAX_RETRIES, _try_idx, traceback.format_exc())
        _log.error("CurvtonDataset: all %d retries exhausted for original idx=%d — "
                   "returning None; collate_fn will replace this slot", _MAX_RETRIES, idx)
        return None


class CombinedCurvtonDataset(Dataset):
    """
    Concatenates CurvtonDataset instances across any subset of
    difficulty × gender combinations from the same local root directory.
    """

    def __init__(self, root_dir: str,
                 difficulties=("easy", "medium", "hard"),
                 genders=("female", "male"),
                 size=512,
                 eval_mode=False):
        self.datasets: list[CurvtonDataset] = []
        for diff in difficulties:
            for gender in genders:
                try:
                    ds = CurvtonDataset(root_dir, diff, gender, size, eval_mode=eval_mode)
                    self.datasets.append(ds)
                except Exception as e:
                    print(f"[CombinedCurvTon] Skipping {diff}/{gender}: {e}")

        self._cum_lengths: list[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cum_lengths.append(total)

        print(f"[CombinedCurvTon] Total triplets: {total}")

    def __len__(self):
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, idx):
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
def get_curvton_dataloaders(root_dir: str, batch_size=8, num_workers=32,
                            size=512, genders=("female", "male")):
    """
    Train dataloaders per difficulty + combined 'all', reading from local disk.
    ``root_dir`` = absolute path to the dataset root, e.g.
    ``"/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"``.

    Returns ``{"easy": DataLoader, "medium": DataLoader,
               "hard": DataLoader, "all": DataLoader}``.
    """
    loaders: dict[str, DataLoader] = {}
    for diff in CurvtonDataset.DIFFICULTIES:
        ds = CombinedCurvtonDataset(root_dir, difficulties=(diff,),
                                    genders=genders, size=size, eval_mode=False)
        loaders[diff] = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True,
                                   collate_fn=collate_fn,
                                   pin_memory=True,
                                   persistent_workers=(num_workers > 0),
                                   prefetch_factor=(4 if num_workers > 0 else None))
        print(f"[Train DataLoader] {diff}: {len(loaders[diff])} batches")

    all_ds = CombinedCurvtonDataset(root_dir,
                                    difficulties=CurvtonDataset.DIFFICULTIES,
                                    genders=genders, size=size, eval_mode=False)
    loaders["all"] = DataLoader(all_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=(num_workers > 0),
                                prefetch_factor=(4 if num_workers > 0 else None))
    print(f"[Train DataLoader] all:  {len(loaders['all'])} batches")
    return loaders


def get_curvton_test_dataloaders(root_dir: str, batch_size=8, num_workers=32,
                                 size=512, genders=("female", "male")):
    """
    Test dataloaders per difficulty + combined 'all', reading from local disk.
    ``root_dir`` = absolute path to the test dataset root.

    shuffle=False, drop_last=False — evaluate on every sample.
    Returns ``{"easy": DataLoader, "medium": DataLoader,
               "hard": DataLoader, "all": DataLoader}``.
    """
    loaders: dict[str, DataLoader] = {}
    for diff in CurvtonDataset.DIFFICULTIES:
        ds = CombinedCurvtonDataset(root_dir, difficulties=(diff,),
                                    genders=genders, size=size, eval_mode=False)
        loaders[diff] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, drop_last=False,
                                   collate_fn=collate_fn,
                                   pin_memory=True,
                                   persistent_workers=(num_workers > 0),
                                   prefetch_factor=(4 if num_workers > 0 else None))
        print(f"[Test DataLoader] {diff}: {len(loaders[diff])} batches")

    all_ds = CombinedCurvtonDataset(root_dir,
                                    difficulties=CurvtonDataset.DIFFICULTIES,
                                    genders=genders, size=size, eval_mode=False)
    loaders["all"] = DataLoader(all_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=False,
                                collate_fn=collate_fn)
    print(f"[Test DataLoader] all:  {len(loaders['all'])} batches")
    return loaders


# ============================================================
# TRIPLET DATASET  (dresscode / viton-hd test sets)
# ============================================================

def _triplet_eval_transform(size: int = 512) -> transforms.Compose:
    """Eval transform for triplet datasets with no crop/resize."""
    _ = size
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

class TripletDataset(Dataset):
    """Load (cloth, person, tryon) triplets from the triplet_dataset layout.

    Directory structure::

        {root_dir}/
            {subset}/
                cloth_image/           *.jpg
                initial_person_image/  *.jpg
                tryon_image/           *.jpg

    Subsets::

        dresscode/dresses
        dresscode/lower_body
        dresscode/upper_body
        viton_hd

    Files are matched by **identical stem** across the three subdirectories.
    Returns a dict with keys ``ground_truth``, ``cloth``, ``person`` —
    the same format used throughout the rest of the codebase.
    """

    SUBSETS = (
        "dresscode/dresses",
        "dresscode/lower_body",
        "dresscode/upper_body",
        "viton_hd",
    )

    def __init__(self, root_dir: str, subset: str, size: int = 512,
                 transform=None, eval_mode=False):
        self.root_dir = root_dir
        self.subset   = subset
        self.size     = size
        self.eval_mode = eval_mode

        cloth_dir  = os.path.join(root_dir, subset, "cloth_image")
        person_dir = os.path.join(root_dir, subset, "initial_person_image")
        tryon_dir  = os.path.join(root_dir, subset, "tryon_image")

        for d in (cloth_dir, person_dir, tryon_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"TripletDataset: expected directory '{d}' does not exist."
                )

        _IMG_EXTS = {".jpg", ".jpeg", ".png"}

        def _stem_map(directory):
            """Return {stem: filename} for image files. If a stem has both
            .jpg and .png, .jpg wins (arbitrary but deterministic)."""
            out = {}
            for f in os.listdir(directory):
                ext = os.path.splitext(f)[1].lower()
                if ext in _IMG_EXTS:
                    stem = os.path.splitext(f)[0]
                    if stem not in out:
                        out[stem] = f
            return out

        cloth_map  = _stem_map(cloth_dir)
        person_map = _stem_map(person_dir)
        tryon_map  = _stem_map(tryon_dir)

        common = sorted(cloth_map.keys() & person_map.keys() & tryon_map.keys())
        if not common:
            raise ValueError(
                f"TripletDataset: no common stems found for subset '{subset}' "
                f"in '{root_dir}'"
            )

        self._items = [
            (
                os.path.join(cloth_dir,  cloth_map[stem]),
                os.path.join(person_dir, person_map[stem]),
                os.path.join(tryon_dir,  tryon_map[stem]),
            )
            for stem in common
        ]

        if transform is not None:
            self._tf = transform
        elif size and size > 0:
            self._tf = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self._tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        _MAX_RETRIES = 20
        for _attempt in range(_MAX_RETRIES):
            _try_idx = idx if _attempt == 0 else random.randint(0, len(self) - 1)
            try:
                cloth_path, person_path, tryon_path = self._items[_try_idx]
                cloth_img = _local_load_image(cloth_path)
                person_img = _local_load_image(person_path)
                tryon_img = _local_load_image(tryon_path)
                if self.eval_mode:
                    person_t, cloth_t, tryon_t = _eval_triplet_person_aware_tensor(
                        person_img, cloth_img, tryon_img, out_size=self.size, pre_resize_size=768
                    )
                else:
                    cloth_t = self._tf(cloth_img)
                    person_t = self._tf(person_img)
                    tryon_t = self._tf(tryon_img)
                return {
                    "ground_truth": tryon_t,
                    "cloth":        cloth_t,
                    "person":       person_t,
                    "mask":         torch.zeros(1, person_t.shape[1], person_t.shape[2]),
                }
            except Exception:
                _log.warning("TripletDataset __getitem__ attempt %d/%d idx=%d failed:\n%s",
                             _attempt + 1, _MAX_RETRIES, _try_idx, traceback.format_exc())
        _log.error("TripletDataset: all %d retries exhausted for original idx=%d — "
                   "returning None; collate_fn will replace this slot", _MAX_RETRIES, idx)
        return None


def get_triplet_test_dataloaders(root_dir: str, batch_size: int = 8,
                                  num_workers: int = 32, size: int = 512):
    """
    Build one DataLoader per subset of the triplet_dataset layout.

    ``root_dir`` = absolute path, e.g.
    ``"/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"``.

    Returns a dict with keys:
        ``"dresscode_dresses"``, ``"dresscode_lower"``, ``"dresscode_upper"``,
        ``"viton_hd"``
    """
    _SUBSET_KEYS: dict[str, str] = {
        "dresscode/dresses":    "dresscode_dresses",
        "dresscode/lower_body": "dresscode_lower",
        "dresscode/upper_body": "dresscode_upper",
        "viton_hd":             "viton_hd",
    }

    loaders: dict[str, DataLoader] = {}
    for subset, key in _SUBSET_KEYS.items():
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.isdir(subset_dir):
            print(f"[TripletDataLoader] subset '{subset}' not found in '{root_dir}', skipping.")
            continue
        _eval_tf = _triplet_eval_transform(size)
        try:
            ds = TripletDataset(root_dir=root_dir, subset=subset, size=size,
                                transform=_eval_tf, eval_mode=True)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[TripletDataLoader] skipping '{subset}': {exc}")
            continue
        loaders[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
        )
        print(f"[Triplet DataLoader] {key}: {len(loaders[key])} batches")
    return loaders


class CombinedTripletDataset(Dataset):
    """
    Concatenate all available TripletDataset subsets from a single root dir.
    Used for Phase-2 training (vitonhd + dresscode as a unified training set).

    ``root_dir`` = e.g. ``"/iopsstor/scratch/cscs/.../triplet_dataset_train"``
    """

    _ALL_SUBSETS = (
        "dresscode/dresses",
        "dresscode/lower_body",
        "dresscode/upper_body",
        "viton_hd",
    )

    def __init__(self, root_dir: str, size: int = 512,
                 subsets: tuple | None = None):
        _use = subsets if subsets is not None else self._ALL_SUBSETS
        self.datasets: list[TripletDataset] = []
        for s in _use:
            subset_dir = os.path.join(root_dir, s)
            if not os.path.isdir(subset_dir):
                _log.warning("CombinedTripletDataset: subset '%s' not found, skipping.", s)
                continue
            try:
                ds = TripletDataset(root_dir=root_dir, subset=s, size=size)
                self.datasets.append(ds)
                _log.info("CombinedTripletDataset: loaded '%s' (%d samples)", s, len(ds))
            except Exception:
                _log.warning("CombinedTripletDataset: error loading '%s':\n%s",
                             s, traceback.format_exc())

        if not self.datasets:
            raise ValueError(
                f"CombinedTripletDataset: no valid subsets found in '{root_dir}'")

        self._cum: list[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cum.append(total)
        _log.info("CombinedTripletDataset: total samples = %d", total)

    def __len__(self) -> int:
        return self._cum[-1]

    def __getitem__(self, idx: int):
        lo, hi = 0, len(self.datasets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        offset = self._cum[lo - 1] if lo > 0 else 0
        return self.datasets[lo][idx - offset]


def get_triplet_train_loader(root_dir: str, batch_size: int = 16,
                              num_workers: int = 32, size: int = 512,
                              world_size: int = 1, rank: int = 0):
    """
    Build a single combined DataLoader for training on triplet_dataset_train
    (all dresscode + viton_hd subsets pooled together).

    Returns ``(DataLoader, DistributedSampler | None)``.
    """
    ds = CombinedTripletDataset(root_dir=root_dir, size=size)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        if world_size > 1 else None
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    _log.info("[Triplet Train Loader] %d batches/epoch (size=%d)", len(loader), size)
    return loader, sampler


# ============================================================
# TRAINING UTILITIES
# ============================================================
def decode_latents(vae, latents, decode_batch_size=1, vae_fp16=True):
    with torch.no_grad():
        chunks = []
        b = latents.shape[0]
        step = max(1, int(decode_batch_size))
        use_amp = (
            vae_fp16
            and latents.device.type == "cuda"
            and latents.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )
        for i in range(0, b, step):
            z = latents[i : i + step]
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = vae.decode(z / 0.18215).sample
            else:
                out = vae.decode(z / 0.18215).sample
            chunks.append(out)
        imgs = torch.cat(chunks, dim=0)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


@torch.no_grad()
def run_full_inference(model, cond_latents, num_inference_steps=30):
    """
    Run complete inference loop starting from pure noise.
    Returns the fully denoised latents.
    """
    device = cond_latents.device

    # Set scheduler to inference mode with fewer steps
    model.scheduler.set_timesteps(num_inference_steps)

    # cond_latents: [B, 4, 64, 128]  = VAE(cat([person, cloth], width))
    # noise starts at the same shape as cond_latents
    latents = torch.randn_like(cond_latents)   # [B, 4, 64, 128]

    # No text encoder — use a fixed zero embedding [B, 77, 768] (CLIP hidden dim)
    B = cond_latents.shape[0]
    text_emb = torch.zeros(B, 77, 768, device=device, dtype=cond_latents.dtype)

    # Iterative denoising loop
    for t in model.scheduler.timesteps:
        # Channel-concat: [noisy(4) ‖ cond(4)] → [B, 8, 64, 128]
        unet_input = torch.cat([latents, cond_latents], dim=1)

        # Predict noise: output is [B, 4, 64, 128]
        noise_pred = model.unet(unet_input, t, text_emb).sample

        # Denoise one step
        latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def log_images(step, batch, model, noisy_latents, noise_pred,
               cond_latents, target_latents, num_inference_steps=30):
    if not hasattr(wandb, "run") or wandb.run is None:
        return

    def to_wandb_img(tensor, caption):
        img = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return wandb.Image(img, caption=caption)

    # Use runtime sample width (full-res safe) instead of fixed IMAGE_SIZE.
    gt_w = int(batch["ground_truth"].shape[-1])

    with torch.no_grad():
        # ── Full inference: run all scheduler steps ──────────────
        print(f"\n🔄 Running full inference ({num_inference_steps} steps)...")
        model.unet.eval()
        full_inference_latents = run_full_inference(model, cond_latents, num_inference_steps)
        model.unet.train()
        print(f"✓ Full inference complete")

        # Decode latents and keep only try-on width based on current batch.
        def _tryon(lat):
            """VAE-decode latent and return try-on crop with dynamic width."""
            decoded = decode_latents(model.vae, lat)
            w = min(gt_w, int(decoded.shape[-1]))
            return decoded[:, :, :, :w]

        full_inference_img   = _tryon(full_inference_latents)    # predicted try-on
        target_img           = _tryon(target_latents)            # GT try-on reconstructed
        noisy_img            = _tryon(noisy_latents)             # noisy target (left half)

        # Single-step noise residual approximation → try-on
        denoised_single      = noisy_latents - noise_pred        # [B,4,64,128]
        denoised_single_img  = _tryon(denoised_single)           # [B,3,512,512]

        # Cond latent decoded (shows person‖cloth context, keep full width)
        cond_img = decode_latents(model.vae, cond_latents)       # [B,3,512,1024]

        gt = (batch['ground_truth'][0:1] + 1) / 2
        cloth = (batch['cloth'][0:1] + 1) / 2
        _pkey = 'person' if 'person' in batch else 'masked_person'
        person_vis = (batch[_pkey][0:1] + 1) / 2

    wandb.log({
        "images/ground_truth": to_wandb_img(gt, "Ground Truth"),
        "images/cloth": to_wandb_img(cloth, "Cloth"),
        "images/person": to_wandb_img(person_vis, "Person (raw or masked)"),
        "images/cond_decoded": to_wandb_img(cond_img, "Cond = VAE(person‖cloth) decoded"),
        "images/target_decoded": to_wandb_img(target_img, "Target Decoded"),
        "images/noisy_decoded": to_wandb_img(noisy_img, "Noisy Decoded"),
        "images/denoised_single_step": to_wandb_img(denoised_single_img, "Denoised (Single Step)"),
        "images/full_inference": to_wandb_img(full_inference_img, f"Full Inference ({num_inference_steps} steps)"),
    }, step=step)


def log_images_distributed(step, batch, model, cond_latents,
                           target_latents, num_inference_steps=50,
                           rank=0, world_size=1):
    """
    Parallel image logging across all GPUs.

    Every rank runs full inference on its **own** first batch sample in
    parallel.  The generated images (plus GT / cloth / person) are gathered
    to rank 0 via ``dist.gather``, which then logs *all* of them (one per
    GPU) to wandb — giving you ``world_size`` diverse samples per logging
    step instead of just one.

    All ranks MUST call this function (it contains collective operations).
    Only rank 0 actually logs to wandb.
    """
    # If W&B is unavailable or not initialized, skip image logging gracefully.
    if not hasattr(wandb, "run") or wandb.run is None:
        return

    is_main = (rank == 0)
    is_dist = (world_size > 1) and dist.is_available() and dist.is_initialized()
    device  = cond_latents.device
    # Use runtime sample width (full-res safe) instead of fixed IMAGE_SIZE.
    gt_w     = int(batch["ground_truth"].shape[-1])

    def _tensor_to_np(tensor):
        """[C,H,W] float [0,1] → [H,W,C] uint8 numpy."""
        return (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # ── 1. Each rank: run full inference on its first sample ─────────
    with torch.no_grad():
        # Take only the first sample from this rank's batch  → [1, ...]
        my_cond   = cond_latents[0:1]      # [1, 4, 64, 128]
        my_target = target_latents[0:1]    # [1, 4, 64, 128]

        model.unet.eval()
        my_pred_latents = run_full_inference(model, my_cond, num_inference_steps)
        model.unet.train()

        # Decode and keep only try-on width based on current batch.
        _pred_dec = decode_latents(model.vae, my_pred_latents)
        _tgt_dec  = decode_latents(model.vae, my_target)
        _w_pred   = min(gt_w, int(_pred_dec.shape[-1]))
        _w_tgt    = min(gt_w, int(_tgt_dec.shape[-1]))
        my_pred_img   = _pred_dec[:, :, :, :_w_pred]
        my_target_img = _tgt_dec[:, :, :, :_w_tgt]

        # Raw inputs  (from batch, in [-1,1] → [0,1])
        my_gt    = (batch['ground_truth'][0:1] + 1) / 2      # [1, 3, H, W]
        my_cloth = (batch['cloth'][0:1] + 1) / 2
        _pkey    = 'person' if 'person' in batch else 'masked_person'
        my_person = (batch[_pkey][0:1] + 1) / 2

        # Move everything to the same device for gather
        my_pred_img   = my_pred_img.to(device)
        my_target_img = my_target_img.to(device)
        my_gt         = my_gt.to(device)
        my_cloth      = my_cloth.to(device)
        my_person     = my_person.to(device)

    # ── 2. Gather all ranks' images to rank 0 ───────────────────────
    def _gather_tensor(tensor):
        """Gather [1, C, H, W] from every rank → list of [1, C, H, W] on rank 0."""
        if not is_dist:
            return [tensor]
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)] if is_main else None
        dist.gather(tensor.contiguous(), gather_list=gather_list, dst=0)
        return gather_list  # only meaningful on rank 0

    gathered_pred   = _gather_tensor(my_pred_img)
    gathered_target = _gather_tensor(my_target_img)
    gathered_gt     = _gather_tensor(my_gt)
    gathered_cloth  = _gather_tensor(my_cloth)
    gathered_person = _gather_tensor(my_person)

    # ── 3. Rank 0: log all images to wandb ──────────────────────────
    if is_main:
        log_payload = {}
        for gpu_idx in range(world_size):
            tag = f"gpu{gpu_idx}"
            log_payload[f"images/{tag}/generated"] = wandb.Image(
                _tensor_to_np(gathered_pred[gpu_idx][0]),
                caption=f"Generated (GPU {gpu_idx}, {num_inference_steps} steps)")
            log_payload[f"images/{tag}/ground_truth"] = wandb.Image(
                _tensor_to_np(gathered_gt[gpu_idx][0]),
                caption=f"Ground Truth (GPU {gpu_idx})")
            log_payload[f"images/{tag}/target_decoded"] = wandb.Image(
                _tensor_to_np(gathered_target[gpu_idx][0]),
                caption=f"Target Decoded (GPU {gpu_idx})")
            log_payload[f"images/{tag}/cloth"] = wandb.Image(
                _tensor_to_np(gathered_cloth[gpu_idx][0]),
                caption=f"Cloth (GPU {gpu_idx})")
            log_payload[f"images/{tag}/person"] = wandb.Image(
                _tensor_to_np(gathered_person[gpu_idx][0]),
                caption=f"Person (GPU {gpu_idx})")

        wandb.log(log_payload, step=step)
        print(f"✓ Logged {world_size} generated images from all GPUs at step {step}")

    # ── 4. Sync all ranks before returning to training ──────────────
    if is_dist:
        dist.barrier()


# ============================================================
# POSE KEYPOINT ERROR HELPER
# ============================================================

def _pose_keypoint_error(img_a_np: np.ndarray, img_b_np: np.ndarray) -> float | None:
    """
    Compute normalised mean L2 keypoint distance between two [H,W,3] uint8 images.

    Pose model: MediaPipe BlazePose  (model_complexity=1, "Full" model, 33 landmarks).
      - model_complexity=0 → Lite  (fastest, least accurate)
      - model_complexity=1 → Full  (balanced; used here)
      - model_complexity=2 → Heavy (most accurate, slowest)

    Only landmarks that are mutually visible (confidence > 0.5 in BOTH images) are
    included.  Distance is normalised by the image diagonal so the result is in
    [0, 1] regardless of resolution.  Lower is better.

    Returns None if:
      - mediapipe is not installed
      - pose detection fails on either image
      - no landmarks are mutually visible
    """
    if not _POSE_AVAILABLE:
        return None
    try:
        H, W = img_a_np.shape[:2]
        diag = float(np.sqrt(H ** 2 + W ** 2))
        with _mp_pose.Pose(static_image_mode=True,
                           model_complexity=1,
                           enable_segmentation=False,
                           min_detection_confidence=0.5) as pose:
            res_a = pose.process(img_a_np)
            res_b = pose.process(img_b_np)

        if res_a.pose_landmarks is None or res_b.pose_landmarks is None:
            return None

        lm_a = res_a.pose_landmarks.landmark   # 33 landmarks
        lm_b = res_b.pose_landmarks.landmark

        dists = []
        for a, b in zip(lm_a, lm_b):
            if a.visibility > 0.5 and b.visibility > 0.5:
                dx = (a.x - b.x) * W
                dy = (a.y - b.y) * H
                dists.append(np.sqrt(dx ** 2 + dy ** 2) / diag)

        return float(np.mean(dists)) if dists else None
    except Exception as e:
        print(f"[Pose] keypoint error computation failed: {e}")
        return None


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def _run_one_eval_sample(
    model, loader, device, _eval_inf_steps, eval_frac, ootd,
    rank, world_size, is_dist, seed,
    lpips_fn, split_name,
):
    """
    Run one evaluation sample: randomly select eval_frac of the dataset
    (using ``seed`` for reproducibility), run inference, and return
    per-image metric means for this sample.

    Returns dict with keys lpips, ssim, psnr, fid, kid_mean, kid_std
    (all floats) on rank 0; returns None on other ranks.
    """
    import torch.distributed as dist
    is_main = (rank == 0)

    dataset = loader.dataset
    bs      = loader.batch_size
    total_dataset_len = len(dataset)
    n_batches = max(1, math.ceil(len(loader) * eval_frac))

    # Random batch selection for this sample (same seed broadcast to all ranks)
    rng = random.Random(seed)
    all_batch_indices = list(range(len(loader)))
    selected = sorted(rng.sample(all_batch_indices, min(n_batches, len(all_batch_indices))))

    # Stride-based rank assignment within the selected batches
    my_batch_indices = selected[rank::world_size]

    # Build a sub-DataLoader for only this rank's samples
    rank_sample_indices = []
    for bi in my_batch_indices:
        start = bi * bs
        end   = min(start + bs, total_dataset_len)
        rank_sample_indices.extend(range(start, end))

    if not rank_sample_indices:
        # No samples for this rank in this sample — return zero tensors for all_reduce
        dummy = torch.zeros(1, device=device)
        if is_dist:
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)  # keep ranks in sync
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
        return None

    rank_subset = torch.utils.data.Subset(dataset, rank_sample_indices)
    rank_loader = DataLoader(
        rank_subset,
        batch_size=bs,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    if is_main:
        fid = FrechetInceptionDistance(feature=64, reset_real_features=True,
                                        normalize=True).to(device)
        kid = KernelInceptionDistance(feature=64, reset_real_features=True,
                                       normalize=True,
                                       subset_size=min(50, len(rank_sample_indices))).to(device)

    lpips_sum = torch.zeros(1, device=device)
    ssim_sum  = torch.zeros(1, device=device)
    psnr_sum  = torch.zeros(1, device=device)
    count     = torch.zeros(1, device=device)

    with torch.no_grad(), _eval_autocast():
        for batch in rank_loader:
            gt         = batch["ground_truth"].to(device)
            cloth      = batch["cloth"].to(device)
            person_img = batch.get("person", batch.get("masked_person")).to(device)

            # Keep eval conditioning consistent with train.py:
            # person_with_pose (masked person replaced by initial person, black-mask semantics)
            # concatenated with cloth along width.
            # No pose required: directly use the person image
            person_with_pose = person_img

            cond_input   = cloth if ootd else torch.cat([person_with_pose, cloth], dim=3)
            cond_latents = model.vae.encode(cond_input).latent_dist.sample() * 0.18215
            pred_latents = run_full_inference(model, cond_latents, _eval_inf_steps)

            pred_wide  = decode_latents(model.vae, pred_latents)
            gt_w = int(gt.shape[-1])
            pred_tryon = pred_wide[:, :, :, : min(gt_w, int(pred_wide.shape[-1]))]
            real_tryon = (gt / 2 + 0.5).clamp(0, 1)

            lp = lpips_fn(pred_tryon * 2 - 1, real_tryon * 2 - 1)
            lpips_sum += lp.mean().detach()
            ssim_sum  += ssim_metric(pred_tryon, real_tryon).detach()
            psnr_sum  += psnr_metric(pred_tryon, real_tryon).detach()
            count     += 1

            if is_main:
                real_u8 = (real_tryon * 255).to(torch.uint8)
                pred_u8 = (pred_tryon * 255).to(torch.uint8)
                fid.update(real_u8, real=True)
                fid.update(pred_u8, real=False)
                kid.update(real_u8, real=True)
                kid.update(pred_u8, real=False)

    if is_dist:
        dist.all_reduce(lpips_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(count,     op=dist.ReduceOp.SUM)

    del ssim_metric, psnr_metric

    if is_main and count.item() > 0:
        n = count.item()
        fid_score         = fid.compute().item()
        kid_mean_v, kid_std_v = kid.compute()
        del fid, kid
        return {
            "lpips":    (lpips_sum / n).item(),
            "ssim":     (ssim_sum  / n).item(),
            "psnr":     (psnr_sum  / n).item(),
            "fid":      fid_score,
            "kid_mean": kid_mean_v.item(),
            "kid_std":  kid_std_v.item(),
        }
    return None


def evaluate_on_test(model, test_loaders, device, num_inference_steps,
                     eval_frac=0.01, ootd=False, rank=0, world_size=1,
                     num_eval_steps=None, n_samples=10):
    """
    Distributed evaluation with bootstrap-style mean ± std metric estimation.

    Runs ``n_samples`` independent evaluations, each on a different random
    ``eval_frac`` subset of the dataset (default: 1%, 10 independent samples).
    Reports mean ± std across the 10 samples for every metric.

    This gives a stable, uncertainty-aware metric estimate without evaluating
    the entire test set.
    """
    import torch.distributed as dist
    is_dist = dist.is_available() and dist.is_initialized()
    is_main = (rank == 0)

    _eval_inf_steps = num_eval_steps if num_eval_steps is not None else min(10, num_inference_steps)

    model.unet.eval()
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    log_dict = {}

    if is_main:
        total_full_images = sum(
            len(loader) * loader.batch_size for loader in test_loaders.values()
        )
        approx_per_sample = sum(
            max(1, math.ceil(len(loader) * eval_frac)) * loader.batch_size
            for loader in test_loaders.values()
        )
        print(f"\n{'='*60}")
        print(f"[Eval] {len(test_loaders)} splits | "
              f"eval_frac={eval_frac*100:.1f}% | "
              f"n_samples={n_samples} | "
              f"~{approx_per_sample:,} images/sample | "
              f"{total_full_images:,} total in dataset")
        print(f"{'='*60}")

    for split_name, loader in test_loaders.items():
        if is_main:
            print(f"\n[Eval] {split_name}: running {n_samples} samples "
                  f"x {eval_frac*100:.1f}% "
                  f"[{_eval_inf_steps} diffusion steps/batch] ...")

        # Collect per-sample metric values
        sample_results: dict[str, list] = {
            "lpips": [], "ssim": [], "psnr": [], "fid": [], "kid_mean": [], "kid_std": []
        }
        t0 = time.time()
        for s_idx in range(n_samples):
            seed = 42 + s_idx * 1000  # deterministic but varied per sample
            result = _run_one_eval_sample(
                model, loader, device, _eval_inf_steps, eval_frac, ootd,
                rank, world_size, is_dist, seed, lpips_fn, split_name,
            )
            if is_main:
                elapsed = time.time() - t0
                eta = (elapsed / (s_idx + 1)) * (n_samples - s_idx - 1)
                if result is not None:
                    for k, v in result.items():
                        sample_results[k].append(v)
                    print(f"  sample {s_idx+1}/{n_samples}  "
                          f"lpips={result['lpips']:.4f}  "
                          f"ssim={result['ssim']:.4f}  "
                          f"psnr={result['psnr']:.2f}  "
                          f"elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)
                else:
                    print(f"  sample {s_idx+1}/{n_samples}  [no batches — skipped]", flush=True)
            torch.cuda.empty_cache()

        if is_main and sample_results["lpips"]:
            for metric_name, values in sample_results.items():
                arr = np.array(values, dtype=np.float64)
                mean_v = float(arr.mean())
                std_v  = float(arr.std()) if len(arr) > 1 else 0.0
                log_dict[f"test/{split_name}/{metric_name}_mean"] = mean_v
                log_dict[f"test/{split_name}/{metric_name}_std"]  = std_v

            print(f"\n  [{split_name}] Results over {len(sample_results['lpips'])} samples:")
            print(f"    LPIPS : {log_dict[f'test/{split_name}/lpips_mean']:.4f} "
                  f"± {log_dict[f'test/{split_name}/lpips_std']:.4f}")
            print(f"    SSIM  : {log_dict[f'test/{split_name}/ssim_mean']:.4f} "
                  f"± {log_dict[f'test/{split_name}/ssim_std']:.4f}")
            print(f"    PSNR  : {log_dict[f'test/{split_name}/psnr_mean']:.2f} "
                  f"± {log_dict[f'test/{split_name}/psnr_std']:.2f} dB")
            print(f"    FID   : {log_dict[f'test/{split_name}/fid_mean']:.2f} "
                  f"± {log_dict[f'test/{split_name}/fid_std']:.2f}")
            print(f"    KID   : {log_dict[f'test/{split_name}/kid_mean_mean']:.4f} "
                  f"± {log_dict[f'test/{split_name}/kid_mean_std']:.4f}")
        elif is_main:
            print(f"   {split_name} | skipped (no valid samples)")

    del lpips_fn
    torch.cuda.empty_cache()
    model.unet.train()
    return log_dict


# ============================================================
# CURRICULUM TRAINING HELPERS
# ============================================================

# Stage schedules: (w_easy, w_medium, w_hard) — normalised at runtime
# Boundaries (with default --stage_steps 7000): 7k, 14k, 21k
_CURRIC_STAGES = [
    (1.0, 0.0, 0.0),   # stage 0 — 100% easy
    (0.6, 0.4, 0.0),   # stage 1 — 60% easy | 40% medium
    (0.3, 0.3, 0.4),   # stage 2 — 30% easy | 30% medium | 40% hard
]
_REVERSE_STAGES = [
    (0.0, 0.0, 1.0),   # stage 0 — 100% hard
    (0.0, 0.4, 0.6),   # stage 1 — 40% medium | 60% hard
    (0.3, 0.3, 0.4),   # stage 2 — 30% easy | 30% medium | 40% hard
]


def curriculum_weights(step: int, curriculum: str, stage_steps: int, hard_pct: float = None):
    """
    Return (w_easy, w_medium, w_hard) for the current training step.

    hard         – hard stage transitions every `stage_steps` steps
    soft         – linearly interpolates between the same stage targets
    reverse      – hard transitions starting from hard → easy
    soft_reverse – linearly interpolates between reverse stage targets
    none         – uniform (1, 1, 1)
    """
    if curriculum == "none":
        return (1.0, 1.0, 1.0)
    
    stages = _CURRIC_STAGES
    if curriculum in ("reverse", "soft_reverse"):
        if hard_pct is not None:
            h0 = hard_pct / 100.0
            h1 = h0 * 0.6
            h2 = h0 * 0.4
            
            s0 = (0.0, 1.0 - h0, h0)
            
            m1 = (1.0 - h1) * 0.6
            e1 = (1.0 - h1) * 0.4
            s1 = (e1, m1, h1)
            
            m2 = (1.0 - h2) * 0.5
            e2 = (1.0 - h2) * 0.5
            s2 = (e2, m2, h2)
            
            stages = [s0, s1, s2]
        else:
            stages = _REVERSE_STAGES
            
    frac = step / max(stage_steps, 1)          # float stage index
    lo   = min(int(frac), len(stages) - 1)
    hi   = min(lo + 1,   len(stages) - 1)
    if curriculum in ("hard", "reverse"):
        return stages[lo]                      # snap to current stage
    # soft / soft_reverse: linearly blend between stage[lo] and stage[hi]
    t  = frac - int(frac)
    we = stages[lo][0] * (1.0 - t) + stages[hi][0] * t
    wm = stages[lo][1] * (1.0 - t) + stages[hi][1] * t
    wh = stages[lo][2] * (1.0 - t) + stages[hi][2] * t
    return (we, wm, wh)


def subsample_dataset(dataset, fraction, seed=42):
    """Return a random Subset of `dataset` containing `fraction` of the samples."""
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    print(f"  ↳ Sub-sampled {k}/{n} samples ({fraction*100:.0f}%)")
    return Subset(dataset, indices)


