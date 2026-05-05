import math
import os
from typing import Tuple

import torch
from torchvision.utils import save_image


_CURRIC_STAGES = [
    (1.0, 0.0, 0.0),
    (0.6, 0.4, 0.0),
    (0.3, 0.3, 0.4),
]

_REVERSE_STAGES = [
    (0.0, 0.0, 1.0),
    (0.0, 0.4, 0.6),
    (0.3, 0.3, 0.4),
]


def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def make_cosine_timestep_weights(num_steps: int, device: torch.device) -> torch.Tensor:
    steps = torch.arange(num_steps, device=device, dtype=torch.float32)
    weights = torch.sin(math.pi * (steps + 0.5) / num_steps)
    return weights / weights.sum()


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(x0: torch.Tensor, t: torch.Tensor, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    x_t = extract(sqrt_ab, t, x0.shape) * x0 + extract(sqrt_1mab, t, x0.shape) * noise
    return x_t, noise


def x0_to_eps(x_t: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor) -> torch.Tensor:
    return (x_t - extract(sqrt_ab, t, x_t.shape) * x0_pred) / extract(sqrt_1mab, t, x_t.shape).clamp(min=1e-8)


def curriculum_weights(step: int, curriculum: str, stage_steps: int) -> Tuple[float, float, float]:
    if curriculum == "none":
        return 1.0, 1.0, 1.0
    stages = _REVERSE_STAGES if curriculum == "reverse" else _CURRIC_STAGES
    frac = step / max(stage_steps, 1)
    lo = min(int(frac), len(stages) - 1)
    hi = min(lo + 1, len(stages) - 1)
    if curriculum == "hard":
        return stages[lo]
    t = frac - int(frac)
    we = stages[lo][0] * (1 - t) + stages[hi][0] * t
    wm = stages[lo][1] * (1 - t) + stages[hi][1] * t
    wh = stages[lo][2] * (1 - t) + stages[hi][2] * t
    return we, wm, wh


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_batch_preview(batch: torch.Tensor, path: str, nrow: int = 4) -> None:
    ensure_dir(os.path.dirname(path))
    vis = (batch.clamp(-1, 1) + 1) * 0.5
    save_image(vis, path, nrow=nrow)


@torch.no_grad()
def sample_ddim_like(
    model: torch.nn.Module,
    shape: Tuple[int, int, int, int],
    timesteps: int,
    sqrt_ab: torch.Tensor,
    sqrt_1mab: torch.Tensor,
    device: torch.device,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    for step in reversed(range(timesteps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x0_pred = model(x, t, cond).clamp(-1, 1)
        if step == 0:
            x = x0_pred
            break
        eps = x0_to_eps(x, x0_pred, t, sqrt_ab, sqrt_1mab)
        ab_prev = sqrt_ab[step - 1] ** 2
        x = math.sqrt(ab_prev) * x0_pred + math.sqrt(max(1.0 - ab_prev.item(), 0.0)) * eps
    return x

