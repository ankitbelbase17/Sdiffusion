import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


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


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 2 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(a) * b)


class RopeSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def _rope_cache(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        pos = torch.arange(seqlen, device=device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1))
        theta = pos * freq.unsqueeze(0)
        cos = torch.cos(theta).to(dtype=dtype)
        sin = torch.sin(theta).to(dtype=dtype)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D/2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x).view(bsz, seqlen, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]
        cos, sin = self._rope_cache(seqlen, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        return self.out(attn)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = RopeSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, mlp_hidden)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(cond).chunk(6, dim=1)
        a = _modulate(self.norm1(x), shift1, scale1)
        a = self.attn(a)
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
class DiTConfig:
    image_size: int = 64
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    in_channels: int = 3
    cond_in_channels: int = 3
    out_channels: Optional[int] = None
    patch_size: int = 2
    hidden_size: int = 1536
    depth: int = 9
    num_heads: int = 24
    mlp_ratio: float = 4.0
    use_dinov2_cond_encoder: bool = True
    dinov2_model_name: str = "facebook/dinov2-base"
    dinov2_image_size: int = 224


class DiT250M(nn.Module):
    """
    ~250M parameter DiT that predicts clean data x0 from x_t, t.
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_height = cfg.image_height if cfg.image_height is not None else cfg.image_size
        self.image_width = cfg.image_width if cfg.image_width is not None else cfg.image_size
        assert self.image_height % cfg.patch_size == 0, "image_height must be divisible by patch_size"
        assert self.image_width % cfg.patch_size == 0, "image_width must be divisible by patch_size"
        self.grid_h = self.image_height // cfg.patch_size
        self.grid_w = self.image_width // cfg.patch_size
        self.num_tokens = self.grid_h * self.grid_w
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.out_channels = cfg.out_channels if cfg.out_channels is not None else cfg.in_channels

        self.patch_embed_noise = nn.Conv2d(
            cfg.in_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.use_dinov2_cond_encoder = cfg.use_dinov2_cond_encoder
        if self.use_dinov2_cond_encoder:
            self.cond_encoder = Dinov2Model.from_pretrained(cfg.dinov2_model_name)
            self.cond_encoder.requires_grad_(False)
            self.cond_encoder.eval()
            self.cond_proj = nn.Linear(self.cond_encoder.config.hidden_size, cfg.hidden_size)
            self.patch_embed_cond = None
        else:
            self.cond_encoder = None
            self.cond_proj = None
            self.patch_embed_cond = nn.Conv2d(
                cfg.cond_in_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
            )
        self.time_embed = TimestepEmbedder(cfg.hidden_size)
        self.input_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, 2 * cfg.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.final = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed_noise.weight)
        nn.init.zeros_(self.patch_embed_noise.bias)
        if self.patch_embed_cond is not None:
            nn.init.xavier_uniform_(self.patch_embed_cond.weight)
            nn.init.zeros_(self.patch_embed_cond.bias)
        if self.cond_proj is not None:
            nn.init.xavier_uniform_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cond_encoder is not None:
            self.cond_encoder.eval()
        return self

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p = self.cfg.patch_size
        c = self.out_channels
        h, w = self.grid_h, self.grid_w
        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, h * p, w * p)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros_like(x_t)
        x_noise = self.patch_embed_noise(x_t)      # [B, D, H', W']
        if self.use_dinov2_cond_encoder:
            cond_resized = F.interpolate(
                cond,
                size=(self.cfg.dinov2_image_size, self.cfg.dinov2_image_size),
                mode="bilinear",
                align_corners=False,
            )
            cond_rgb = (cond_resized + 1.0) * 0.5
            mean = cond_rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = cond_rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            cond_rgb = (cond_rgb - mean) / std
            cond_out = self.cond_encoder(pixel_values=cond_rgb).last_hidden_state  # [B, 1+N, Dd]
            cond_tokens = cond_out[:, 1:, :]
            n_tokens = cond_tokens.shape[1]
            grid_side = int(math.sqrt(n_tokens))
            cond_map = cond_tokens.transpose(1, 2).reshape(
                cond_tokens.shape[0], cond_tokens.shape[2], grid_side, grid_side
            )
            cond_map = F.interpolate(
                cond_map,
                size=(self.grid_h, self.grid_w),
                mode="bilinear",
                align_corners=False,
            )
            cond_tokens = cond_map.flatten(2).transpose(1, 2)
            cond_tokens = self.cond_proj(cond_tokens)
            x_cond = cond_tokens.transpose(1, 2).reshape(
                cond_tokens.shape[0], self.cfg.hidden_size, self.grid_h, self.grid_w
            )
        else:
            x_cond = self.patch_embed_cond(cond)       # [B, D, H', W']
        x = torch.cat([x_noise, x_cond], dim=3)    # spatial concat on width
        x = x.flatten(2).transpose(1, 2)           # [B, 2N, D]
        t = self.time_embed(timesteps)
        shift, scale = self.input_mod(t).chunk(2, dim=1)
        x = _modulate(x, shift, scale)
        for block in self.blocks:
            x = block(x, t)
        x = self.final(x, t)
        x = x.view(x.shape[0], self.grid_h, self.grid_w * 2, self.cfg.patch_size, self.cfg.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(x.shape[0], self.out_channels, self.image_height, self.image_width * 2)
        return x[:, :, :, : self.image_width]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
