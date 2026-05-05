
"""
VITON-HD Full Training - Stable Diffusion 1.5
Training script with configurable trainable parameters
"""

import os
import io
import glob
import math
import collections
import statistics
import random
import argparse
import numpy as np
from PIL import Image
import boto3
from botocore.config import Config as BotoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
import wandb
import weave

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
    from config import (
        WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY,
        MODEL_NAME, IMAGE_SIZE,
    )
except Exception:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Stable_diffusion")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))

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
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Mask transforms (nearest neighbor to preserve binary values)
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
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
            mask_img = Image.new('L', (self.size, self.size), 0)
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
            'ground_truth': person_tensor,      # [3, H, W] normalized to [-1, 1]
            'cloth': cloth_tensor,              # [3, H, W] normalized to [-1, 1]
            'mask': mask_tensor,                # [1, H, W] in range [0, 1]
            'masked_person': masked_person,     # [3, H, W] normalized to [-1, 1]
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
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError("Empty batch")
    out = {
        'ground_truth': torch.stack([b['ground_truth'] for b in batch]),
        'cloth':        torch.stack([b['cloth']        for b in batch]),
        'mask':         torch.stack([b['mask']         for b in batch]),
    }
    # CurvTon uses 'person' (raw initial image); VitonHD uses 'masked_person'
    if 'person' in batch[0]:
        out['person'] = torch.stack([b['person'] for b in batch])
    else:
        out['masked_person'] = torch.stack([b['masked_person'] for b in batch])
    return out


# ============================================================
# CURVTON DATASET  (reads directly from AWS S3)
# ============================================================
def _make_s3_client():
    """Create a boto3 S3 client using credentials from config.py."""
    try:
        from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    except Exception:
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _s3_list_keys(s3_client, bucket: str, prefix: str, suffix: str = "") -> list[str]:
    """Return all object keys under `prefix` that end with `suffix`."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(suffix):
                keys.append(k)
    return keys


def _s3_load_image(s3_client, bucket: str, key: str) -> Image.Image:
    """Download an S3 object and return a PIL Image."""
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(resp["Body"].read())).convert("RGB")


class CurvtonDataset(Dataset):
    """
    CurvTon Dataset — streams images directly from S3.

    S3 layout (one difficulty + gender leaf)::

        {bucket}/{difficulty}/{gender}/
            cloth_image/   {base}_cloth_{id}.png
            initial_image/ {base}_person.png
            try_on_image/  {base}_vton.png

    ``bucket`` may be a plain bucket name (``"curvton-dataset"``)
    or an ``s3://`` URI (``"s3://curvton-dataset"``).

    Returns dict with keys:
        ground_truth  – try-on result   [3,H,W]  normalised [-1,1]
        cloth         – garment image   [3,H,W]
        mask          – zeros           [1,H,W]
        person        – initial_image   [3,H,W]  (NOT masked)
    """

    DIFFICULTIES = ("easy", "medium", "hard")
    GENDERS      = ("female", "male")

    def __init__(self, bucket: str, difficulty="easy", gender="female", size=512):
        # Normalise bucket: strip s3:// scheme and any trailing path
        self.bucket = bucket.replace("s3://", "").split("/")[0]
        self.size   = size
        leaf_prefix = f"{difficulty}/{gender}/"   # S3 key prefix for this leaf

        # ── index triplets at init time (one-off S3 LIST calls) ──
        s3 = _make_s3_client()

        person_prefix = f"{leaf_prefix}initial_image/"
        cloth_prefix  = f"{leaf_prefix}cloth_image/"
        vton_prefix   = f"{leaf_prefix}try_on_image/"

        print(f"[CurvTon-{difficulty}/{gender}] Listing S3 objects …")
        person_keys = _s3_list_keys(s3, self.bucket, person_prefix, "_person.png")
        cloth_keys  = _s3_list_keys(s3, self.bucket, cloth_prefix,  ".png")
        vton_keys   = _s3_list_keys(s3, self.bucket, vton_prefix,   "_vton.png")

        # Build lookup dicts: base_key -> S3 full key
        cloth_index: dict[str, str] = {}
        for k in cloth_keys:
            fname = k.rsplit("/", 1)[-1]
            if "_cloth_" in fname:
                base = fname.split("_cloth_")[0]   # ends with "_edit"
                cloth_index[base] = k

        vton_index: dict[str, str] = {
            k.rsplit("/", 1)[-1][: -len("_vton.png")]: k
            for k in vton_keys
        }

        # Build (person_key, cloth_key, vton_key) triplets
        self.triplets: list[tuple[str, str, str]] = []
        missing = 0
        for pk in person_keys:
            fname = pk.rsplit("/", 1)[-1]
            base  = fname[: -len("_person.png")]
            ck    = cloth_index.get(base)
            vk    = vton_index.get(base)
            if ck and vk:
                self.triplets.append((pk, ck, vk))
            else:
                missing += 1

        if missing:
            print(f"[CurvTon-{difficulty}/{gender}] Skipped {missing} incomplete triplets")
        print(f"[CurvTon-{difficulty}/{gender}] {len(self.triplets)} valid triplets")

        # Image transform
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self._zero_mask = torch.zeros(1, size, size)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        person_key, cloth_key, vton_key = self.triplets[idx]
        # Create S3 client per-call so DataLoader workers (separate processes)
        # each get their own connection.
        s3 = _make_s3_client()
        try:
            person = self.img_tf(_s3_load_image(s3, self.bucket, person_key))
            cloth  = self.img_tf(_s3_load_image(s3, self.bucket, cloth_key))
            vton   = self.img_tf(_s3_load_image(s3, self.bucket, vton_key))
        except Exception as e:
            print(f"[CurvTon] S3 fetch failed for triplet {idx}: {e}")
            return None

        return {
            "ground_truth": vton,             # [3,H,W] try-on result
            "cloth":        cloth,            # [3,H,W] garment flat-lay
            "mask":         self._zero_mask,  # [1,H,W] zeros
            "person":       person,           # [3,H,W] raw initial_image
        }


class CombinedCurvtonDataset(Dataset):
    """
    Concatenates CurvtonDataset instances across any subset of
    difficulty × gender combinations from the same S3 bucket.
    """

    def __init__(self, bucket: str,
                 difficulties=("easy", "medium", "hard"),
                 genders=("female", "male"),
                 size=512):
        self.datasets: list[CurvtonDataset] = []
        for diff in difficulties:
            for gender in genders:
                try:
                    ds = CurvtonDataset(bucket, diff, gender, size)
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


def get_curvton_dataloaders(bucket: str, batch_size=8, num_workers=4,
                            size=512, genders=("female", "male")):
    """
    Train dataloaders per difficulty + combined 'all', reading from S3.
    ``bucket`` = ``"curvton-dataset"`` or ``"s3://curvton-dataset"``.

    Returns ``{"easy": DataLoader, "medium": DataLoader,
               "hard": DataLoader, "all": DataLoader}``.
    """
    loaders: dict[str, DataLoader] = {}
    for diff in CurvtonDataset.DIFFICULTIES:
        ds = CombinedCurvtonDataset(bucket, difficulties=(diff,),
                                    genders=genders, size=size)
        loaders[diff] = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True,
                                   collate_fn=collate_fn,
                                   pin_memory=True,
                                   persistent_workers=(num_workers > 0),
                                   prefetch_factor=(4 if num_workers > 0 else None))
        print(f"[Train DataLoader] {diff}: {len(loaders[diff])} batches")

    all_ds = CombinedCurvtonDataset(bucket,
                                    difficulties=CurvtonDataset.DIFFICULTIES,
                                    genders=genders, size=size)
    loaders["all"] = DataLoader(all_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=(num_workers > 0),
                                prefetch_factor=(4 if num_workers > 0 else None))
    print(f"[Train DataLoader] all:  {len(loaders['all'])} batches")
    return loaders


def get_curvton_test_dataloaders(bucket: str, batch_size=8, num_workers=4,
                                 size=512, genders=("female", "male")):
    """
    Test dataloaders per difficulty + combined 'all', reading from S3.
    ``bucket`` = ``"curvton-test-dataset"`` or ``"s3://curvton-test-dataset"``.

    shuffle=False, drop_last=False — evaluate on every sample.
    Returns ``{"easy": DataLoader, "medium": DataLoader,
               "hard": DataLoader, "all": DataLoader}``.
    """
    loaders: dict[str, DataLoader] = {}
    for diff in CurvtonDataset.DIFFICULTIES:
        ds = CombinedCurvtonDataset(bucket, difficulties=(diff,),
                                    genders=genders, size=size)
        loaders[diff] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, drop_last=False,
                                   collate_fn=collate_fn,
                                   pin_memory=True,
                                   persistent_workers=(num_workers > 0),
                                   prefetch_factor=(4 if num_workers > 0 else None))
        print(f"[Test DataLoader] {diff}: {len(loaders[diff])} batches")

    all_ds = CombinedCurvtonDataset(bucket,
                                    difficulties=CurvtonDataset.DIFFICULTIES,
                                    genders=genders, size=size)
    loaders["all"] = DataLoader(all_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=False,
                                collate_fn=collate_fn)
    print(f"[Test DataLoader] all:  {len(loaders['all'])} batches")
    return loaders
# MODEL
# ============================================================
class SDModel:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

        # Freeze VAE
        self.vae.requires_grad_(False)
        # UNet input = cat([noisy_latent(4) ‖ cond_latent(4)], dim=1)
        # where cond_latent = VAE(cat([person, cloth], dim=3))  [B,4,64,128]
        old_conv = self.unet.conv_in                    # Conv2d(4, 320, kernel=3, padding=1)
        new_conv = nn.Conv2d(8, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :4] = old_conv.weight        # preserve noise-channel weights
            nn.init.xavier_uniform_(new_conv.weight[:, 4:]) # init cond channels
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        self.unet.config["in_channels"] = 8
        print("✓ Model loaded (8-ch UNet: 4 noise + 4 cond, channel-concat; "
              "cond = VAE(cat([person, cloth], width)))")
    
    def to(self, device):
        self.vae.to(device)
        self.unet.to(device)
        return self

def count_parameters(model, trainable_only=True):
    """Count total and trainable parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def freeze_non_attention(unet):
    """Freeze all parameters except self-attention layers"""
    # First freeze everything
    for param in unet.parameters():
        param.requires_grad = False
    
    # Unfreeze only attention layers
    attention_modules = []
    for name, module in unet.named_modules():
        # Self-attention layers in UNet
        if 'attn1' in name or 'attn2' in name:
            for param in module.parameters():
                param.requires_grad = True
            attention_modules.append(name)
    
    print(f"✓ Unfroze {len(attention_modules)} attention modules")
    return unet

def print_trainable_params(model, mode):
    """Print detailed trainable parameters"""
    print("\n" + "="*60)
    print(f"TRAINABLE PARAMETERS ({mode})")
    print("="*60)
    
    total_params = count_parameters(model.unet, trainable_only=False)
    trainable_params = count_parameters(model.unet, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    print(f"\nUNet Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Trainable ratio:      {100*trainable_params/total_params:.2f}%")
    
    # Group by layer type
    print(f"\nTrainable layers breakdown:")
    layer_counts = {}
    for name, param in model.unet.named_parameters():
        if param.requires_grad:
            # Get layer type from name
            parts = name.split('.')
            layer_type = '.'.join(parts[:3]) if len(parts) > 3 else name
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += param.numel()
    
    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {layer}: {count:,}")
    
    if len(layer_counts) > 20:
        print(f"    ... and {len(layer_counts) - 20} more layers")
    
    print("="*60 + "\n")
    
    return trainable_params

# ============================================================
# TRAINING UTILITIES
# ============================================================
def decode_latents(vae, latents):
    with torch.no_grad():
        imgs = vae.decode(latents / 0.18215).sample
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

def log_images(step, batch, model, noisy_latents, noise_pred, cond_latents, target_latents, num_inference_steps=30):
    def to_wandb_img(tensor, caption):
        img = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return wandb.Image(img, caption=caption)
    
    W = IMAGE_SIZE  # pixel width of one image (512); decoded wide images are 2W

    with torch.no_grad():
        # ── Full inference: run all scheduler steps ──────────────
        print(f"\n🔄 Running full inference ({num_inference_steps} steps)...")
        model.unet.eval()
        full_inference_latents = run_full_inference(model, cond_latents, num_inference_steps)
        model.unet.train()
        print(f"✓ Full inference complete")

        # Decode all wide latents [B,4,64,128] → [B,3,512,1024], slice left=tryon
        def _tryon(lat):
            """VAE-decode a [B,4,64,128] latent and return the left try-on half [B,3,512,512]."""
            return decode_latents(model.vae, lat)[:, :, :, :W]

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
def evaluate_on_test(model, test_loaders, device, num_inference_steps, eval_frac=0.10):
    """
    Compute LPIPS, SSIM, PSNR, FID, KID on `eval_frac` (default 10%) of
    each difficulty split (easy / medium / hard / all) in test_loaders.

    Inference:
        cond_lat   = VAE(cat([person, cloth], dim=W))    [B,4,64,128]
        pred_lat   = run_full_inference(cond_lat)        [B,4,64,128]
        pred_wide  = VAE_decode(pred_lat)                [B,3,512,1024]
        pred_tryon = pred_wide[:,:,:,:512]               [B,3,512,512]  left half
        real_tryon = (ground_truth + 1) / 2             [B,3,512,512]

    Returns a flat dict of "test/{split}/{metric}" values for wandb.log.
    """
    model.unet.eval()
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

    pose_model_desc = (
        "MediaPipe BlazePose model_complexity=1 (Full, 33 landmarks)"
        if _POSE_AVAILABLE else "unavailable — mediapipe not installed"
    )
    print(f"[Eval] Pose model for PKE: {pose_model_desc}")

    log_dict = {}

    for split_name, loader in test_loaders.items():
        n_batches = max(1, math.ceil(len(loader) * eval_frac))
        print(f"\n[Eval] {split_name}: {n_batches}/{len(loader)} batches "
              f"({eval_frac*100:.0f}% of test set) ...")

        lpips_vals, ssim_vals, psnr_vals = [], [], []
        pke_vals: list[float] = []   # pose keypoint error (may stay empty)

        fid        = FrechetInceptionDistance(feature=2048, reset_real_features=True,
                                              normalize=True).to(device)
        kid        = KernelInceptionDistance(feature=2048, reset_real_features=True,
                                             normalize=True, subset_size=50).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            gt     = batch["ground_truth"].to(device)   # [B,3,H,W]  in [-1,1]
            cloth  = batch["cloth"].to(device)
            # CurvTon: 'person' = raw initial_image; VitonHD: 'masked_person'
            person_img = batch["person"].to(device)

            # Cond: cat([person, cloth], width) → [B,3,512,1024] → VAE → [B,4,64,128]
            cond_input   = torch.cat([person_img, cloth], dim=3)
            cond_latents = model.vae.encode(cond_input).latent_dist.sample() * 0.18215

            # Full denoising inference → [B,4,64,128]
            pred_latents = run_full_inference(model, cond_latents, num_inference_steps)

            # Decode → [B,3,512,1024] → left half = try-on [B,3,512,512]
            pred_wide  = decode_latents(model.vae, pred_latents)
            pred_tryon = pred_wide[:, :, :, :IMAGE_SIZE]

            real_tryon = (gt / 2 + 0.5).clamp(0, 1)                   # [B,3,H,W]

            # LPIPS (expects input in [-1,1])
            lp = lpips_fn(pred_tryon * 2 - 1, real_tryon * 2 - 1)
            lpips_vals.extend(lp.view(-1).cpu().tolist())

            # SSIM & PSNR
            ssim_vals.append(ssim_metric(pred_tryon, real_tryon).item())
            psnr_vals.append(psnr_metric(pred_tryon, real_tryon).item())

            # FID / KID require [0,255] uint8
            real_u8 = (real_tryon * 255).to(torch.uint8)
            pred_u8 = (pred_tryon * 255).to(torch.uint8)
            fid.update(real_u8, real=True)
            fid.update(pred_u8, real=False)
            kid.update(real_u8, real=True)
            kid.update(pred_u8, real=False)

            # Pose keypoint error: per-image, CPU numpy, graceful fallback
            if _POSE_AVAILABLE:
                for b_idx in range(real_tryon.shape[0]):
                    real_np = (real_tryon[b_idx].permute(1, 2, 0).cpu().numpy() * 255
                               ).clip(0, 255).astype(np.uint8)
                    pred_np = (pred_tryon[b_idx].permute(1, 2, 0).cpu().numpy() * 255
                               ).clip(0, 255).astype(np.uint8)
                    pke = _pose_keypoint_error(real_np, pred_np)
                    if pke is not None:
                        pke_vals.append(pke)

        fid_score         = fid.compute().item()
        kid_mean, kid_std = kid.compute()

        lpips_arr = np.array(lpips_vals)
        ssim_arr  = np.array(ssim_vals)
        psnr_arr  = np.array(psnr_vals)

        log_dict[f"test/{split_name}/lpips_mean"] = float(lpips_arr.mean())
        log_dict[f"test/{split_name}/lpips_std"]  = float(lpips_arr.std(ddof=1) if len(lpips_arr) > 1 else 0.0)
        log_dict[f"test/{split_name}/ssim_mean"]  = float(ssim_arr.mean())
        log_dict[f"test/{split_name}/ssim_std"]   = float(ssim_arr.std(ddof=1) if len(ssim_arr) > 1 else 0.0)
        log_dict[f"test/{split_name}/psnr_mean"]  = float(psnr_arr.mean())
        log_dict[f"test/{split_name}/psnr_std"]   = float(psnr_arr.std(ddof=1) if len(psnr_arr) > 1 else 0.0)
        log_dict[f"test/{split_name}/fid"]        = fid_score
        log_dict[f"test/{split_name}/kid_mean"]   = kid_mean.item()
        log_dict[f"test/{split_name}/kid_std"]    = kid_std.item()

        # Pose keypoint error (only logged when at least one valid detection)
        if pke_vals:
            pke_arr = np.array(pke_vals)
            log_dict[f"test/{split_name}/pke_mean"] = float(pke_arr.mean())
            log_dict[f"test/{split_name}/pke_std"]  = float(pke_arr.std(ddof=1) if len(pke_arr) > 1 else 0.0)
            pke_str = (f"  PKE={log_dict[f'test/{split_name}/pke_mean']:.4f}"
                       f"\u00b1{log_dict[f'test/{split_name}/pke_std']:.4f}")
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
    model.unet.train()
    return log_dict


# ============================================================
# CURRICULUM TRAINING HELPERS
# ============================================================

# Stage schedules: (w_easy, w_medium, w_hard) — normalised at runtime
# Boundaries (with default --stage_steps 7000): 7k, 14k, 21k
_CURRIC_STAGES = [
    (1.0, 0.0, 0.0),   # stage 0 — easy only
    (0.3, 0.7, 0.0),   # stage 1 — mostly medium
    (0.2, 0.3, 0.5),   # stage 2 — hard-skewed
    (0.2, 0.3, 0.4),   # stage 3 — 20% easy | 30% medium | 40% hard
]
_REVERSE_STAGES = [
    (0.0, 0.0, 1.0),   # stage 0 — hard only
    (0.0, 0.3, 0.7),   # stage 1 — mostly hard
    (0.2, 0.3, 0.5),   # stage 2 — same end-state as forward curriculum
    (0.34, 0.33, 0.33), # stage 3 — fully balanced (mirror of easy-only)
]


def _curriculum_weights(step: int, curriculum: str, stage_steps: int):
    """
    Return (w_easy, w_medium, w_hard) for the current training step.

    hard    – hard stage transitions every `stage_steps` steps
    soft    – linearly interpolates between the same stage targets
    reverse – hard transitions starting from hard → easy
    none    – uniform (1, 1, 1)
    """
    if curriculum == "none":
        return (1.0, 1.0, 1.0)
    stages = _REVERSE_STAGES if curriculum == "reverse" else _CURRIC_STAGES
    frac = step / max(stage_steps, 1)          # float stage index
    lo   = min(int(frac), len(stages) - 1)
    hi   = min(lo + 1,   len(stages) - 1)
    if curriculum in ("hard", "reverse"):
        return stages[lo]                      # snap to current stage
    # soft: linearly blend between stage[lo] and stage[hi]
    t  = frac - int(frac)
    we = stages[lo][0] * (1.0 - t) + stages[hi][0] * t
    wm = stages[lo][1] * (1.0 - t) + stages[hi][1] * t
    wh = stages[lo][2] * (1.0 - t) + stages[hi][2] * t
    return (we, wm, wh)


def _subsample_dataset(dataset, fraction, seed=42):
    """Return a random Subset of `dataset` containing `fraction` of the samples."""
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    print(f"  ↳ Sub-sampled {k}/{n} samples ({fraction*100:.0f}%)")
    return Subset(dataset, indices)


# ============================================================
# MAIN TRAINING
# ============================================================
def train(args):
    wandb.login(key=WANDB_API_KEY)
    weave.init(f"{os.getenv('WANDB_ENTITY', WANDB_ENTITY)}/{os.getenv('WANDB_PROJECT', WANDB_PROJECT)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── GPU performance flags ───────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True          # auto-tune convolution kernels
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on Ampere+ for matmul
        torch.backends.cudnn.allow_tf32 = True          # TF32 for cuDNN convolutions
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
    
    # Set trainable parameters based on mode
    if args.train_mode == "attention_only":
        model.unet = freeze_non_attention(model.unet)
        run_name = f"{args.dataset}_attention_only_bs{args.batch_size}_{args.curriculum}"
    else:
        run_name = f"{args.dataset}_full_unet_bs{args.batch_size}_{args.curriculum}"

    experiments_root = "/iopsstor/scratch/cscs/dbartaula/experiments_assets"
    if args.checkpoint_dir == ".":
        args.checkpoint_dir = os.path.join(experiments_root, run_name, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Print trainable parameters
    trainable_params = print_trainable_params(model, args.train_mode)

    # torch.compile (PyTorch 2.0+) — speeds up training via graph compilation
    if hasattr(torch, "compile"):
        print("\n⚙  Compiling UNet with torch.compile (mode='reduce-overhead') ...")
        model.unet = torch.compile(model.unet, mode="reduce-overhead")
        print("   ✓ torch.compile applied to UNet.")
    else:
        print("\n⚠  torch.compile not available (requires PyTorch >= 2.0). Skipping.")

    # Dataset & DataLoader
    diff_loaders: dict = {}   # per-difficulty loaders — built for curriculum sampling
    if use_curvton:
        genders = ("female", "male") if args.gender == "all" else (args.gender,)

        # Build one DataLoader per difficulty (needed regardless of curriculum mode)
        for _diff in CurvtonDataset.DIFFICULTIES:
            _ds = CombinedCurvtonDataset(
                bucket=args.curvton_data_path,
                difficulties=(_diff,),
                genders=genders,
                size=IMAGE_SIZE,
            )
            _ds = _subsample_dataset(_ds, args.data_fraction)
            diff_loaders[_diff] = DataLoader(
                _ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=(4 if args.num_workers > 0 else None),
            )
            print(f"✓ CurvTon [{_diff}] DataLoader: {len(diff_loaders[_diff])} batches/epoch")

        # "All" combined loader — used for non-curriculum and batches_per_epoch reference
        if args.difficulty == "all":
            difficulties = CurvtonDataset.DIFFICULTIES
        else:
            difficulties = (args.difficulty,)
        train_dataset = CombinedCurvtonDataset(
            bucket=args.curvton_data_path,
            difficulties=difficulties,
            genders=genders,
            size=IMAGE_SIZE,
        )
        train_dataset = _subsample_dataset(train_dataset, args.data_fraction)
        _frac_tag = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
        dataset_label = f"CurvTon-{args.difficulty}-{args.gender}-{args.curriculum}{_frac_tag}"
    else:
        train_dataset = VitonHDDataset(
            root_dir=args.viton_data_path,
            split='train',
            size=IMAGE_SIZE,
        )
        train_dataset = _subsample_dataset(train_dataset, args.data_fraction)
        _frac_tag = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
        dataset_label = f"VITON-HD{_frac_tag}"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    batches_per_epoch = len(train_loader)
    print(f"✓ DataLoader ({dataset_label}): {batches_per_epoch} batches per epoch")

    # ── Test loaders for periodic evaluation ────────────────────
    test_loaders = None
    if args.curvton_test_data_path:
        genders_test = ("female", "male") if args.gender == "all" else (args.gender,)
        print(f"\nBuilding CurvTon test loaders from {args.curvton_test_data_path} ...")
        test_loaders = get_curvton_test_dataloaders(
            bucket=args.curvton_test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
            genders=genders_test,
        )
        print(f"✓ Test loaders built: {list(test_loaders.keys())}")
    else:
        print("\n[Eval] No --curvton_test_data_path provided; test evaluation will be skipped.")
    # ─────────────────────────────────────────────────────────────

    # WandB
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", WANDB_PROJECT),
        entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model": MODEL_NAME,
            "train_mode": args.train_mode,
            "dataset": dataset_label,
            "trainable_params": trainable_params,
            "curriculum": args.curriculum,
            "stage_steps": args.stage_steps,
            "data_fraction": args.data_fraction,
            "pose_model": "MediaPipe BlazePose model_complexity=1 (Full, 33 landmarks)",
            "lr_schedule": f"cosine_anneal_{args.lr:.0e}_to_1e-05",
            "lr_eta_min": 1e-5,
        },
        name=run_name
    )
    
    # Optimizer (only trainable params)
    trainable_params_list = [p for p in model.unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=args.lr)
    scaler = GradScaler()

    # Cosine annealing LR: starts at args.lr, anneals to 1e-5 over all training steps
    _total_steps = args.epochs * len(train_loader)
    _LR_ETA_MIN  = 1e-5
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(_total_steps, 1), eta_min=_LR_ETA_MIN)

    global_step = 0
    start_epoch = 0
    ema_loss    = None      # may be overwritten by checkpoint

    # ── Checkpoint Resume Logic ──────────────────────────────────
    ckpt_to_load = None
    if args.resume:
        # Explicit path supplied via --resume
        ckpt_to_load = args.resume
        print(f"\n▶ --resume specified: {ckpt_to_load}")
    else:
        # Auto-detect latest checkpoint in checkpoint_dir
        pattern = os.path.join(
            args.checkpoint_dir,
            f"checkpoint_vitonhd_{args.train_mode}_step_*.pt"
        )
        candidates = glob.glob(pattern)
        if candidates:
            def _step_num(p):
                try:
                    return int(os.path.basename(p).split("_step_")[1].replace(".pt", ""))
                except Exception:
                    return -1
            ckpt_to_load = max(candidates, key=_step_num)
            print(f"\n▶ Auto-detected latest checkpoint: {ckpt_to_load}")

    if ckpt_to_load:
        print(f"   Loading checkpoint …")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"   ✓ LR scheduler state restored.")
        else:
            # Fast-forward scheduler to match completed steps
            for _ in range(global_step):
                lr_scheduler.step()
            print(f"   ⚠ No scheduler state in checkpoint — fast-forwarded {global_step} steps.")
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"   ✓ GradScaler state restored.")
        if "ema_loss" in ckpt and ckpt["ema_loss"] is not None:
            ema_loss = ckpt["ema_loss"]
            print(f"   ✓ EMA loss restored: {ema_loss:.6f}")
        print(f"   ✓ Resumed — global_step={global_step}, start_epoch={start_epoch}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print("\n✓ No checkpoint found. Starting fresh with Xavier init on new UNet channels.")
    # ─────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print(f"TRAINING: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Mode: {args.train_mode}, Batch size: {args.batch_size}")
    print(f"LR Schedule: Cosine annealing — {args.lr:.0e} → {_LR_ETA_MIN:.0e} over {_total_steps} steps")
    print("="*60 + "\n")

    # ── Pre-allocate reusable tensors ───────────────────────────
    _cached_text_emb = torch.zeros(args.batch_size, 77, 768, device=device)

    # ── Running stats (window of 100 steps) ─────────────────────
    _WINDOW = 100
    loss_window      = collections.deque(maxlen=_WINDOW)
    grad_norm_window = collections.deque(maxlen=_WINDOW)
    if not isinstance(ema_loss, (float, int)):   # may have been set by checkpoint resume
        ema_loss = None
    EMA_DECAY        = 0.99
    # ────────────────────────────────────────────────────────────
    
    for epoch in range(start_epoch, args.epochs):
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
            if use_curvton and args.curriculum != "none":
                we, wm, wh = _curriculum_weights(global_step, args.curriculum, args.stage_steps)
                _tw = we + wm + wh
                _chosen = random.choices(
                    list(CurvtonDataset.DIFFICULTIES),
                    weights=[we / _tw, wm / _tw, wh / _tw],
                )[0]
                batch = _next_curriculum_batch(_chosen)
            else:
                batch = _item

            gt         = batch['ground_truth'].to(device, non_blocking=True)
            cloth      = batch['cloth'].to(device, non_blocking=True)
            # CurvTon: 'person' = raw initial_image (no masking)
            # VitonHD: 'masked_person' = person with cloth area greyed out
            person_img = batch.get('person', batch.get('masked_person')).to(device, non_blocking=True)

            # Fused VAE encode: batch cond + target into a single forward pass
            # cond   = cat([person, cloth], width) → [B,3,512,1024]
            # target = cat([gt,     cloth], width) → [B,3,512,1024]
            # Stack along batch → [2B,3,512,1024] → VAE → [2B,4,64,128] → split
            with torch.no_grad(), autocast():
                B = gt.shape[0]
                cond_input   = torch.cat([person_img, cloth], dim=3)  # [B,3,512,1024]
                target_input = torch.cat([gt,         cloth], dim=3)  # [B,3,512,1024]
                fused_input  = torch.cat([cond_input, target_input], dim=0)  # [2B,3,512,1024]
                fused_latents = model.vae.encode(fused_input).latent_dist.sample() * 0.18215
                cond_latents   = fused_latents[:B]   # [B,4,64,128]
                target_latents = fused_latents[B:]   # [B,4,64,128]

            # Diffusion (all shapes are [B,4,64,128])
            noise     = torch.randn_like(target_latents)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps,
                                      (target_latents.shape[0],), device=device).long()
            noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)

            # UNet input: channel-concat [noisy(4) ‖ cond(4)] → [B,8,64,128]
            unet_input = torch.cat([noisy_latents, cond_latents], dim=1)

            text_emb = _cached_text_emb[:B]

            # Forward
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                # UNet output: [B,4,64,128] — full tensor used for loss
                noise_pred = model.unet(unet_input, timesteps, text_emb).sample
                loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            scaler.scale(loss).backward()
            # Unscale before grad-norm so we measure the true (unscaled) gradient magnitude
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1e9).item()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # ── Update running stats ────────────────────────────
            loss_val = loss.item()
            loss_window.append(loss_val)
            grad_norm_window.append(grad_norm)

            # EMA loss
            ema_loss = loss_val if ema_loss is None else EMA_DECAY * ema_loss + (1 - EMA_DECAY) * loss_val

            # Window mean & variance (population variance)
            loss_mean = statistics.mean(loss_window)
            loss_var  = statistics.pvariance(loss_window) if len(loss_window) > 1 else 0.0
            gn_mean   = statistics.mean(grad_norm_window)
            gn_var    = statistics.pvariance(grad_norm_window) if len(grad_norm_window) > 1 else 0.0
            # ───────────────────────────────────────────────────

            current_lr = optimizer.param_groups[0]['lr']
            epoch_losses.append(loss_val)
            wandb.log({
                "train/loss":              loss_val,
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
            if use_curvton and args.curriculum != "none":
                _cwe, _cwm, _cwh = _curriculum_weights(global_step, args.curriculum, args.stage_steps)
                wandb.log({
                    "curriculum/w_easy":   _cwe,
                    "curriculum/w_medium": _cwm,
                    "curriculum/w_hard":   _cwh,
                    "curriculum/stage":    min(global_step // max(args.stage_steps, 1),
                                               len(_CURRIC_STAGES) - 1),
                }, step=global_step)

            # Log images with full inference
            if global_step % args.image_log_interval == 0:
                log_images(global_step, batch, model, noisy_latents, noise_pred, cond_latents, target_latents, args.num_inference_steps)
            
            # Save checkpoint
            if global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(
                    args.checkpoint_dir,
                    f"checkpoint_vitonhd_{args.train_mode}_step_{global_step}.pt",
                )
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

            # ── Periodic test-set evaluation ────────────────────
            if (test_loaders is not None
                    and global_step > 0
                    and global_step % args.eval_interval == 0):
                print(f"\n📊 Running test-set evaluation at step {global_step} ...")
                eval_metrics = evaluate_on_test(
                    model, test_loaders, device,
                    num_inference_steps=args.num_inference_steps,
                    eval_frac=0.10,
                )
                wandb.log(eval_metrics, step=global_step)
                print(f"✓ Eval metrics logged to W&B")
            
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            global_step += 1
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        wandb.log({"train/epoch_avg_loss": avg_loss}, step=global_step)
        print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_loss:.6f}")
    
    # Final save
    final_path = os.path.join(args.checkpoint_dir, f"checkpoint_vitonhd_{args.train_mode}_final.pt")
    torch.save({
        'step': global_step,
        'epoch': args.epochs,
        'train_mode': args.train_mode,
        'unet_state_dict': model.unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_loss': ema_loss,
    }, final_path)
    
    print("\n" + "="*60)
    print(f"✓ TRAINING COMPLETE! Total steps: {global_step}")
    print(f"✓ Final checkpoint: {final_path}")
    print("="*60)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Try-On Training (VITON-HD / CurvTon)")
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
                        choices=["easy", "medium", "hard", "all"],
                        help="CurvTon difficulty split to train on (default: all)")
    parser.add_argument("--gender", type=str, default="all",
                        choices=["female", "male", "all"],
                        help="CurvTon gender subset to train on (default: all)")
    # ── Common ───────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of DataLoader workers")
    # ── Curriculum ────────────────────────────────────────────────
    parser.add_argument("--curriculum", type=str, default="none",
                        choices=["none", "hard", "soft", "reverse"],
                        help="Curriculum strategy: none | hard | soft | reverse")
    parser.add_argument("--stage_steps", type=int, default=10000,
                        help="Steps per curriculum stage (hard / soft / reverse)")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use, e.g. 0.1 for 10%% (default: 1.0 = all)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--image_log_interval", type=int, default=250, help="Log images every N steps")
    parser.add_argument("--eval_interval", type=int, default=2500,
                        help="Run test-set evaluation every N steps (default: 2500)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for full denoising")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a specific checkpoint .pt file to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default=".",
                        help="Directory to scan for the latest checkpoint when --resume is not given")
    
    args = parser.parse_args()

    # Validate that the required data path was supplied
    if args.dataset == "curvton" and not args.curvton_data_path:
        parser.error("--curvton_data_path is required when --dataset curvton")
    if args.dataset == "vitonhd" and not args.viton_data_path:
        parser.error("--viton_data_path is required when --dataset vitonhd")

    train(args)
