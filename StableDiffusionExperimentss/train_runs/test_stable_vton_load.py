#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _setup_paths():
    root = Path(__file__).resolve().parents[1]
    cross_arch = root / "cross-architecture"
    for p in (root, cross_arch):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _init_dist(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_dist = world_size > 1

    if use_dist:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
    return use_dist, rank, world_size, local_rank, device


def _finish_dist(use_dist: bool):
    if use_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Smoke-test StableVTON model loading")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-rank batch size. If omitted, derived from --global_batch_size/world_size.")
    parser.add_argument("--global_batch_size", type=int, default=4,
                        help="Global batch size across all ranks.")
    args = parser.parse_args()

    _setup_paths()
    use_dist, rank, world_size, local_rank, device = _init_dist(args.device)
    from StableVTON.train_stable_vton_local import StableVTONModel  # noqa: E402

    if rank == 0:
        print(f"[INFO] world_size={world_size} local_rank={local_rank} device={device}")
        print(f"[INFO] loading StableVTONModel with model_name={args.model_name}")
    model = StableVTONModel(args.model_name).to(device)
    model.train()

    if args.batch_size is not None:
        b = args.batch_size
    else:
        b = max(1, args.global_batch_size // world_size)
    if rank == 0:
        print(f"[INFO] global_batch_size={args.global_batch_size} per_rank_batch_size={b} world_size={world_size}")
    h = 64
    w = 64
    noisy = torch.randn(b, 4, h, w, device=device)
    mask_lat = torch.zeros(b, 1, h, w, device=device)
    agnostic_lat = torch.randn(b, 4, h, w, device=device)
    pose_lat = torch.randn(b, 4, h, w, device=device)
    timesteps = torch.randint(
        0, model.scheduler.config.num_train_timesteps, (b,), device=device
    ).long()

    pred = model(noisy, mask_lat, agnostic_lat, pose_lat, timesteps)
    target = torch.randn_like(pred)
    loss = (pred - target).pow(2).mean()
    loss.backward()

    if rank == 0:
        print("[OK] StableVTONModel loaded successfully.")
        print("[OK] Components: vae, scheduler, unet(13-channel conv_in)")
        print(f"[OK] multi-gpu forward+backward passed. rank0 loss={loss.item():.6f}")

    _finish_dist(use_dist)


if __name__ == "__main__":
    main()
