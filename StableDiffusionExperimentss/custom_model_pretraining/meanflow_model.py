import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_size: int = 256) -> None:
        super().__init__()
        self.freq_size = freq_size
        self.proj = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.freq_size // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.freq_size % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.proj(emb)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(cond).chunk(6, dim=1)
        a = _modulate(self.norm1(x), shift1, scale1)
        a, _ = self.attn(a, a, a, need_weights=False)
        x = x + gate1.unsqueeze(1) * a
        m = self.mlp(_modulate(self.norm2(x), shift2, scale2))
        x = x + gate2.unsqueeze(1) * m
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond).chunk(2, dim=1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


@dataclass
class MeanFlowDiTConfig:
    image_size: int = 64
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    in_channels: int = 3
    out_channels: Optional[int] = None
    patch_size: int = 2
    hidden_size: int = 1536
    depth: int = 9
    num_heads: int = 24
    mlp_ratio: float = 4.0


class MeanFlowDiT250M(nn.Module):
    """
    DiT backbone conditioned on (t, r) for average-velocity prediction u_theta(z_t, r, t).
    """

    def __init__(self, cfg: MeanFlowDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_height = cfg.image_height if cfg.image_height is not None else cfg.image_size
        self.image_width = cfg.image_width if cfg.image_width is not None else cfg.image_size
        assert self.image_height % cfg.patch_size == 0, "image_height must be divisible by patch_size"
        assert self.image_width % cfg.patch_size == 0, "image_width must be divisible by patch_size"
        self.grid_h = self.image_height // cfg.patch_size
        self.grid_w = self.image_width // cfg.patch_size
        self.num_tokens = self.grid_h * self.grid_w
        self.out_channels = cfg.out_channels if cfg.out_channels is not None else cfg.in_channels

        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, cfg.hidden_size))
        self.t_embed = TimestepEmbedder(cfg.hidden_size)
        self.r_embed = TimestepEmbedder(cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.final = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p = self.cfg.patch_size
        c = self.out_channels
        h, w = self.grid_h, self.grid_w
        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, h * p, w * p)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(z_t)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cond = self.t_embed(t) + self.r_embed(r)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final(x, cond)
        return self.unpatchify(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
