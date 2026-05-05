"""Shared local training utilities for cross-architecture VTON experiments.

The files in this directory intentionally keep the model code local. They use
the same CurvTON batch contract as the rest of this workspace:
ground_truth, cloth, person/masked_person tensors in [-1, 1].
"""

import argparse
import glob
import os
import sys
import datetime
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from config import IMAGE_SIZE  # noqa: E402
except Exception:
    IMAGE_SIZE = (512, 384)
from utils import CombinedCurvtonDataset, CurvtonDataset, collate_fn  # noqa: E402


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main: bool


def setup_dist() -> DistInfo:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
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
    return DistInfo(rank, local_rank, world_size, device, rank == 0)


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_curvton_loader(args, dist_info):
    genders = ("female", "male") if args.gender == "all" else (args.gender,)
    difficulty_map = {
        "all": CurvtonDataset.DIFFICULTIES,
        "easy_hard": ("easy", "hard"),
        "medium_hard": ("medium", "hard"),
    }
    difficulties = difficulty_map.get(args.difficulty, (args.difficulty,))
    image_size = IMAGE_SIZE if getattr(args, "image_size", None) is None else args.image_size
    dataset = CombinedCurvtonDataset(
        root_dir=args.curvton_data_path,
        difficulties=difficulties,
        genders=genders,
        size=image_size,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_info.world_size,
        rank=dist_info.rank,
        shuffle=True,
        drop_last=True,
    ) if dist_info.world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    return loader, sampler


def batch_images(batch, device, dtype=torch.float32):
    gt = batch["ground_truth"].to(device, dtype=dtype, non_blocking=True)
    cloth = batch["cloth"].to(device, dtype=dtype, non_blocking=True)
    person = batch["person"].to(
        device, dtype=dtype, non_blocking=True
    )
    return person, cloth, gt


def wrap_ddp(module, dist_info, find_unused=False):
    module = module.to(dist_info.device)
    if dist_info.world_size > 1:
        return DDP(
            module,
            device_ids=[dist_info.local_rank],
            output_device=dist_info.local_rank,
            find_unused_parameters=find_unused,
        )
    return module


def save_checkpoint(path, model, optimizer, step, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def latest_checkpoint(run_dir: str):
    final_ckpt = os.path.join(run_dir, "ckpt_final.pt")
    if os.path.exists(final_ckpt):
        return final_ckpt
    candidates = glob.glob(os.path.join(run_dir, "ckpt_*.pt")) + glob.glob(
        os.path.join(run_dir, "ckpt_step_*.pt")
    )
    if not candidates:
        return None

    def _step_num(p):
        base = os.path.basename(p)
        try:
            if base.startswith("ckpt_step_"):
                return int(base.split("ckpt_step_")[1].split(".pt")[0])
            return int(base.split("ckpt_")[1].split(".pt")[0])
        except Exception:
            return -1

    return max(candidates, key=_step_num)


def latest_stage_checkpoint(run_dir: str, prefix: str):
    final_ckpt = os.path.join(run_dir, f"{prefix}_final.pt")
    if os.path.exists(final_ckpt):
        return final_ckpt
    candidates = glob.glob(os.path.join(run_dir, f"{prefix}_*.pt"))
    if not candidates:
        return None

    def _step_num(p):
        base = os.path.basename(p)
        try:
            return int(base.split(f"{prefix}_")[1].split(".pt")[0])
        except Exception:
            return -1

    return max(candidates, key=_step_num)


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--curvton_data_path", type=str, required=True)
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "all", "easy_hard", "medium_hard"])
    parser.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--image_log_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="/iopsstor/scratch/cscs/dbartaula/experiments_assets")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="Stable_diffusion")
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image resize target. Use 0 for full/native resolution. "
             "If omitted, uses IMAGE_SIZE default.",
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetGenerator(nn.Module):
    """Compact UNet used by CP-VTON TOM and local cross-architecture training."""

    def __init__(self, in_channels, out_channels=3, base=64):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 8)
        self.d3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = F.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return torch.tanh(self.out(d1))


def tv_loss(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (
        x[:, :, :, 1:] - x[:, :, :, :-1]
    ).abs().mean()
