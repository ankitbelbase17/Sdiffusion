import os
import re
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


_FC_MC_RE = re.compile(r"_(?:fc|mc)_")


def _maybe_init_wandb(args, is_main: bool):
    if not is_main or getattr(args, "disable_wandb", False) or wandb is None:
        return None
    try:
        return wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    except Exception:
        return None


def _to_wandb_image(batch: torch.Tensor, caption: str):
    if wandb is None:
        return None
    vis = (batch.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(vis, nrow=min(4, vis.shape[0]))
    return wandb.Image(grid, caption=caption)


def _resolve_first_existing(base: str, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    return None


class StableCategoryMaskPoseDataset(Dataset):
    """
    Expected structure per category/gender:
      cloth_image/, initial_person_image/, tryon_image/,
      mask_image/, and one pose dir among:
      densepose_image|densepose|pose_image|pose|image_densepose
    """

    def __init__(self, root_dir: str, category: str = "all", gender: str = "all", size: int = 0):
        self.root_dir = root_dir
        self.category = category.lower()
        self.gender = gender.lower()
        self.size = int(size) if size and int(size) > 0 else 0
        self.samples = []

        self.img_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.mask_tf = transforms.ToTensor()

        categories = ("dresses", "upper_body", "lower_body", "uncertain") if self.category == "all" else (self.category,)
        genders = ("female", "male") if self.gender == "all" else (self.gender,)
        for cat in categories:
            for g in genders:
                self._collect(cat, g)

        if not self.samples:
            raise RuntimeError(f"No samples found under {root_dir}")

    def _collect(self, category: str, gender: str):
        leaf = os.path.join(self.root_dir, category, gender)
        cloth_dir = os.path.join(leaf, "cloth_image")
        person_dir = os.path.join(leaf, "initial_person_image")
        tryon_dir = os.path.join(leaf, "tryon_image")
        mask_dir = os.path.join(leaf, "mask_image")
        pose_dir = _resolve_first_existing(
            leaf,
            ["densepose_image", "densepose", "pose_image", "pose", "image_densepose"],
        )
        if pose_dir is None:
            raise FileNotFoundError(f"Missing pose directory under: {leaf}")

        for d in (cloth_dir, person_dir, tryon_dir, mask_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")

        cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".png")])
        person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(".png")}
        tryon_set = {f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")}
        mask_set = {f for f in os.listdir(mask_dir) if f.lower().endswith(".png")}
        pose_set = {f for f in os.listdir(pose_dir) if f.lower().endswith(".png")}

        for fname in cloth_files:
            if fname not in tryon_set or fname not in mask_set or fname not in pose_set:
                continue
            stem = os.path.splitext(fname)[0]
            m = _FC_MC_RE.search(stem)
            if m is None:
                continue
            person_stem = stem[: m.start()]
            if person_stem not in person_stems:
                continue
            self.samples.append(
                (
                    os.path.join(cloth_dir, fname),
                    os.path.join(person_dir, person_stem + ".png"),
                    os.path.join(mask_dir, fname),
                    os.path.join(pose_dir, fname),
                    os.path.join(tryon_dir, fname),
                    category,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cloth_p, person_p, mask_p, pose_p, tryon_p, cat = self.samples[idx]
        cloth = self.img_tf(Image.open(cloth_p).convert("RGB"))
        person = self.img_tf(Image.open(person_p).convert("RGB"))
        pose = self.img_tf(Image.open(pose_p).convert("RGB"))
        mask = (self.mask_tf(Image.open(mask_p).convert("L")) > 0.5).float()
        target = self.img_tf(Image.open(tryon_p).convert("RGB"))
        return {
            "cloth": cloth,
            "person": person,
            "pose": pose,
            "mask": mask,
            "ground_truth": target,
            "category": cat,
        }


def _collate(rows):
    return {
        "cloth": torch.stack([r["cloth"] for r in rows], dim=0),
        "person": torch.stack([r["person"] for r in rows], dim=0),
        "pose": torch.stack([r["pose"] for r in rows], dim=0),
        "mask": torch.stack([r["mask"] for r in rows], dim=0),
        "ground_truth": torch.stack([r["ground_truth"] for r in rows], dim=0),
        "category": [r["category"] for r in rows],
    }

