import argparse
import glob
import datetime
import os
import random
from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader import build_curvton_difficulty_files, build_phase2_triplet_files, make_loader, subset_files
from meanflow_model import MeanFlowDiT250M, MeanFlowDiTConfig, count_parameters
from utils import curriculum_weights, ensure_dir, save_batch_preview

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _maybe_init_ddp() -> Tuple[int, int, int, bool, torch.device]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    is_main = rank == 0
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
    return rank, local_rank, world_size, is_main, device


def _next_from(
    iters: Dict[str, Iterator[torch.Tensor]],
    loaders: Dict[str, torch.utils.data.DataLoader],
    key: str,
) -> torch.Tensor:
    try:
        return next(iters[key])
    except StopIteration:
        iters[key] = iter(loaders[key])
        return next(iters[key])


def _maybe_init_wandb(args: argparse.Namespace, is_main: bool):
    if not is_main or args.disable_wandb:
        return None
    if os.environ.get("DISABLE_WANDB", "0") == "1":
        return None
    if wandb is None:
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            settings=wandb.Settings(start_method="thread"),
        )
    except Exception:
        return None


def _to_wandb_image(batch: torch.Tensor, caption: str):
    if wandb is None:
        return None
    vis = (batch.clamp(-1, 1) + 1) * 0.5
    grid = make_grid(vis, nrow=4)
    return wandb.Image(grid, caption=caption)


def _catvton_wide_tensors(batch: dict, device: torch.device):
    gt = batch["ground_truth"].to(device, non_blocking=True)
    cloth = batch["cloth"].to(device, non_blocking=True)
    person = batch["person"].to(device, non_blocking=True)
    cond_input = torch.cat([person, cloth], dim=3)   # person || cloth
    target_input = torch.cat([gt, cloth], dim=3)     # gt || cloth
    return cond_input, target_input


def _jvp_directional_total_derivative(
    model: torch.nn.Module,
    z_t: torch.Tensor,
    t_model: torch.Tensor,
    r_model: torch.Tensor,
    v_cond: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes:
      u_theta(z_t, t, r)
      d/dt u_theta(z_t, t, r) along dz/dt = v_cond

    Uses JVP with direction (v_cond, 1, 0) over inputs (z_t, t, r).
    """
    z_req = z_t.detach().requires_grad_(True)
    t_req = t_model.detach().requires_grad_(True)
    r_req = r_model.detach()
    direction_z = v_cond.detach()
    direction_t = torch.ones_like(t_req)
    direction_r = torch.zeros_like(r_req)

    def _f(inp_z: torch.Tensor, inp_t: torch.Tensor, inp_r: torch.Tensor) -> torch.Tensor:
        return model(inp_z, inp_t, inp_r)

    u_pred, du_dt_total = torch.autograd.functional.jvp(
        _f,
        (z_req, t_req, r_req),
        (direction_z, direction_t, direction_r),
        create_graph=False,
        strict=False,
    )
    return u_pred, du_dt_total


def train(args: argparse.Namespace) -> None:
    rank, _, world_size, is_main, device = _maybe_init_ddp()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    image_height = args.image_size
    image_width = args.image_width if args.image_width > 0 else image_height * 2
    cfg = MeanFlowDiTConfig(
        image_size=args.image_size,
        image_height=image_height,
        image_width=image_width,
        in_channels=3,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    model = MeanFlowDiT250M(cfg).to(device)
    raw_model = model
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    if is_main:
        params_m = count_parameters(raw_model) / 1e6
        print(
            f"Resolved MeanFlow custom DiT 250M hidden={args.hidden_size} depth={args.depth} heads={args.num_heads}"
        )
        print(f"MeanFlow DiT params: {params_m:.2f}M")

    wb_run = _maybe_init_wandb(args, is_main)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    diff_files = build_curvton_difficulty_files(args.data_path, gender=args.gender)
    for k in diff_files:
        diff_files[k] = subset_files(diff_files[k], args.data_fraction, seed=args.seed)

    diff_loaders: Dict[str, torch.utils.data.DataLoader] = {}
    for diff in ("easy", "medium", "hard"):
        loader, _ = make_loader(
            diff_files[diff],
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )
        diff_loaders[diff] = loader

    all_files = diff_files["easy"] + diff_files["medium"] + diff_files["hard"]
    all_loader, _ = make_loader(
        all_files,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=True,
    )

    phase2_loader = None
    phase2_iter = None
    if args.phase2_data_path:
        phase2_files = subset_files(
            build_phase2_triplet_files(args.phase2_data_path, fallback_to_self_pairs=True),
            args.data_fraction,
            seed=args.seed + 7,
        )
        phase2_loader, _ = make_loader(
            phase2_files,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )

    iters = {k: iter(v) for k, v in diff_loaders.items()}
    all_iter = iter(all_loader)

    run_root = os.path.join(args.output_dir, args.run_name)
    ckpt_dir = os.path.join(run_root, "checkpoints")
    sample_dir = os.path.join(run_root, "samples")
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt")) + glob.glob(
            os.path.join(ckpt_dir, "ckpt_final.pt")
        )
        if candidates:
            def _step_num(path: str):
                base = os.path.basename(path)
                if base == "ckpt_final.pt":
                    return float("inf")
                try:
                    return int(base.split("ckpt_step_")[1].split(".pt")[0])
                except Exception:
                    return -1

            ckpt_to_load = max(candidates, key=_step_num)

    global_step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=device)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        global_step = int(ckpt.get("step", 0))

    pbar = tqdm(total=args.max_steps, disable=not is_main, desc="meanflow-training")
    pbar.update(global_step)
    while global_step < args.max_steps:
        in_phase2 = phase2_loader is not None and global_step >= args.phase2_start_step
        if in_phase2:
            if phase2_iter is None:
                phase2_iter = iter(phase2_loader)
            try:
                phase2_batch = next(phase2_iter)
            except StopIteration:
                phase2_iter = iter(phase2_loader)
                phase2_batch = next(phase2_iter)
            cond_vis, x_data = _catvton_wide_tensors(phase2_batch, device)
        elif args.curriculum == "none":
            try:
                batch = next(all_iter)
            except StopIteration:
                all_iter = iter(all_loader)
                batch = next(all_iter)
            cond_vis, x_data = _catvton_wide_tensors(batch, device)
        else:
            we, wm, wh = curriculum_weights(global_step, args.curriculum, args.stage_steps)
            diff = random.choices(["easy", "medium", "hard"], weights=[we, wm, wh])[0]
            batch = _next_from(iters, diff_loaders, diff)
            cond_vis, x_data = _catvton_wide_tensors(batch, device)
        eps = torch.randn_like(x_data)

        # Sample interval [r, t]; default is one-step interval with r fixed to 0.
        r = torch.full((x_data.shape[0],), args.r_min, device=device)
        t = torch.rand((x_data.shape[0],), device=device) * (1.0 - r) + r
        z_t = (1.0 - t.view(-1, 1, 1, 1)) * x_data + t.view(-1, 1, 1, 1) * eps
        v_cond = eps - x_data

        t_model = t * args.time_embed_scale
        r_model = r * args.time_embed_scale

        with autocast(enabled=(device.type == "cuda")):
            u_pred, du_dt_total = _jvp_directional_total_derivative(model, z_t, t_model, r_model, v_cond)
            delta = (t - r).view(-1, 1, 1, 1)
            u_target = v_cond - delta * du_dt_total
            loss = F.mse_loss(u_pred, u_target.detach())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if is_main and global_step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if wb_run is not None:
                wb_run.log({"train/loss": float(loss.item()), "global_step": global_step}, step=global_step)

        if is_main and global_step % args.image_log_interval == 0:
            # MeanFlow one-step inference sample from pure noise: z0 = z1 - u(z1, 0, 1)
            with torch.no_grad():
                z1 = torch.randn_like(x_data)
                r_inf = torch.zeros((x_data.shape[0],), device=device)
                t_inf = torch.ones((x_data.shape[0],), device=device)
                u_inf = (raw_model if world_size == 1 else model.module)(
                    z1,
                    t_inf * args.time_embed_scale,
                    r_inf * args.time_embed_scale,
                )
                z0_inf = (z1 - u_inf).detach().cpu()

            z_r_pred = (z_t - (t - r).view(-1, 1, 1, 1) * u_pred).detach().cpu()
            pred_tryon = z_r_pred[:8, :, :, :image_height]
            inf_tryon = z0_inf[:8, :, :, :image_height]
            target_tryon = x_data[:8].detach().cpu()[:, :, :, :image_height]
            save_batch_preview(inf_tryon, os.path.join(sample_dir, f"meanflow_infer_step_{global_step}.png"))
            if wb_run is not None:
                pred_img = _to_wandb_image(pred_tryon, f"MeanFlow train prediction step {global_step}")
                infer_img = _to_wandb_image(inf_tryon, f"MeanFlow one-step inference step {global_step}")
                gt_img = _to_wandb_image(target_tryon, f"Target tryon step {global_step}")
                cond_img = _to_wandb_image(cond_vis[:8].detach().cpu(), f"Condition person||cloth step {global_step}")
                log_payload = {"global_step": global_step}
                if pred_img is not None:
                    log_payload["images/meanflow_prediction"] = pred_img
                if infer_img is not None:
                    log_payload["images/meanflow_inference_tryon"] = infer_img
                if gt_img is not None:
                    log_payload["images/target_tryon"] = gt_img
                if cond_img is not None:
                    log_payload["images/condition_concat"] = cond_img
                wb_run.log(log_payload, step=global_step)

        if is_main and global_step > 0 and global_step % args.save_interval == 0:
            to_save = raw_model.state_dict() if world_size == 1 else model.module.state_dict()
            torch.save(
                {"step": global_step, "model": to_save, "cfg": cfg.__dict__},
                os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt"),
            )

        global_step += 1
        pbar.update(1)

    if is_main:
        to_save = raw_model.state_dict() if world_size == 1 else model.module.state_dict()
        torch.save({"step": global_step, "model": to_save, "cfg": cfg.__dict__}, os.path.join(ckpt_dir, "ckpt_final.pt"))
        if wb_run is not None:
            wb_run.finish()
    if world_size > 1:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Custom 250M DiT MeanFlow-style pretraining")
    p.add_argument("--run_name", type=str, default="custom_dit_meanflow_pretrain")
    p.add_argument("--data_path", type=str, required=True, help="CurvTon root containing easy/medium/hard")
    p.add_argument("--phase2_data_path", type=str, default=None, help="Optional final-stage dataset path")
    p.add_argument("--phase2_start_step", type=int, default=28801)
    p.add_argument("--output_dir", type=str, default="/iopsstor/scratch/cscs/dbartaula/experiments_assets")
    p.add_argument("--curriculum", type=str, default="soft", choices=["none", "soft", "reverse", "hard"])
    p.add_argument("--stage_steps", type=int, default=4000)
    p.add_argument("--max_steps", type=int, default=12000)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--image_log_interval", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=1280)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--num_heads", type=int, default=20)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_width", type=int, default=-1, help="Target width; <=0 means 2x image_size (CATVTON concat).")
    p.add_argument("--data_fraction", type=float, default=1.0)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no_resume", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="Stable_diffusion")
    p.add_argument("--disable_wandb", action="store_true", default=False)
    p.add_argument("--r_min", type=float, default=0.0, help="Lower bound for interval start r.")
    p.add_argument("--time_embed_scale", type=float, default=1000.0, help="Scale for continuous times passed to timestep embedding.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
