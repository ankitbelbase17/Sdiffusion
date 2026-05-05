import os
import random
import re
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_FC_MC_RE = re.compile(r"_(?:fc|mc)_")
_SUBSETS = ("dresscode_dresses", "dresscode_lower", "dresscode_upper", "viton_hd")


def _list_images(root: str) -> List[str]:
    files: List[str] = []
    for d, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(_IMG_EXTS):
                files.append(os.path.join(d, n))
    return files


def _stem_map(dir_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in os.listdir(dir_path):
        if n.lower().endswith(_IMG_EXTS):
            out[os.path.splitext(n)[0]] = n
    return out


def _build_curvton_triplets(root_dir: str, diff: str, gender: str) -> List[Tuple[str, str, str]]:
    leaf = os.path.join(root_dir, diff, gender)
    cloth_dir = os.path.join(leaf, "cloth_image")
    person_dir = os.path.join(leaf, "initial_person_image")
    tryon_dir = os.path.join(leaf, "tryon_image")
    if not (os.path.isdir(cloth_dir) and os.path.isdir(person_dir) and os.path.isdir(tryon_dir)):
        return []

    cloth_files = sorted(f for f in os.listdir(cloth_dir) if f.lower().endswith(_IMG_EXTS))
    person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(_IMG_EXTS)}
    tryon_stems = {os.path.splitext(f)[0] for f in os.listdir(tryon_dir) if f.lower().endswith(_IMG_EXTS)}

    triplets: List[Tuple[str, str, str]] = []
    for fname in cloth_files:
        stem = os.path.splitext(fname)[0]
        m = _FC_MC_RE.search(stem)
        if m is None:
            continue
        person_base = stem[: m.start()]
        if person_base not in person_stems or stem not in tryon_stems:
            continue
        # Preserve actual extension by lookup.
        person_path = None
        for ext in _IMG_EXTS:
            p = os.path.join(person_dir, person_base + ext)
            if os.path.exists(p):
                person_path = p
                break
        tryon_path = None
        for ext in _IMG_EXTS:
            p = os.path.join(tryon_dir, stem + ext)
            if os.path.exists(p):
                tryon_path = p
                break
        if person_path is None or tryon_path is None:
            continue
        triplets.append((os.path.join(cloth_dir, fname), person_path, tryon_path))
    return triplets


def _build_triplet_dataset_triplets(root_dir: str) -> List[Tuple[str, str, str]]:
    triplets: List[Tuple[str, str, str]] = []
    for subset in _SUBSETS:
        subset_dir = os.path.join(root_dir, subset)
        cloth_dir = os.path.join(subset_dir, "cloth_image")
        person_dir = os.path.join(subset_dir, "initial_person_image")
        tryon_dir = os.path.join(subset_dir, "tryon_image")
        if not (os.path.isdir(cloth_dir) and os.path.isdir(person_dir) and os.path.isdir(tryon_dir)):
            continue
        cloth_map = _stem_map(cloth_dir)
        person_map = _stem_map(person_dir)
        tryon_map = _stem_map(tryon_dir)
        for stem in sorted(cloth_map.keys() & person_map.keys() & tryon_map.keys()):
            triplets.append(
                (
                    os.path.join(cloth_dir, cloth_map[stem]),
                    os.path.join(person_dir, person_map[stem]),
                    os.path.join(tryon_dir, tryon_map[stem]),
                )
            )
    return triplets


class TripletImageDataset(Dataset):
    """
    Returns dict with keys: ground_truth, cloth, person, mask.
    """

    def __init__(self, triplets: Iterable[Tuple[str, str, str]], image_size: int = 64) -> None:
        self.triplets = list(triplets)
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self._zero_mask = torch.zeros(1, image_size, image_size)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        cloth_path, person_path, tryon_path = self.triplets[idx]
        cloth = self.tf(Image.open(cloth_path).convert("RGB"))
        person = self.tf(Image.open(person_path).convert("RGB"))
        gt = self.tf(Image.open(tryon_path).convert("RGB"))
        return {
            "ground_truth": gt,
            "cloth": cloth,
            "person": person,
            "mask": self._zero_mask,
        }


def build_curvton_difficulty_files(root_dir: str, gender: str = "all") -> Dict[str, List[Tuple[str, str, str]]]:
    files_by_diff: Dict[str, List[Tuple[str, str, str]]] = {}
    genders: Tuple[str, ...] = ("female", "male") if gender == "all" else (gender,)
    for diff in ("easy", "medium", "hard"):
        collected: List[Tuple[str, str, str]] = []
        for g in genders:
            collected.extend(_build_curvton_triplets(root_dir, diff, g))
        files_by_diff[diff] = collected
    return files_by_diff


def build_phase2_triplet_files(root_dir: str, fallback_to_self_pairs: bool = True) -> List[Tuple[str, str, str]]:
    files = _build_triplet_dataset_triplets(root_dir)
    if files:
        return files
    if not fallback_to_self_pairs:
        return []
    # Fallback for arbitrary image roots: use the same image as cloth/person/gt
    # so training can continue even if phase2 root is plain image-only.
    imgs = _list_images(root_dir)
    return [(p, p, p) for p in imgs]


def subset_files(files: List, fraction: float, seed: int = 42) -> List:
    if fraction >= 1.0:
        return files
    n = len(files)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    return [files[i] for i in indices]


def make_loader(
    files: List[Tuple[str, str, str]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int = 1,
    rank: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler | None]:
    ds = TripletImageDataset(files, image_size=image_size)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler

