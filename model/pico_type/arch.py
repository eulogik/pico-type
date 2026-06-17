"""pico-type: a tiny byte-level multi-head content classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TIERS: Dict[str, int] = {
    "tiny": 16,
    "small": 64,
    "base": 192,
    "pro": 576,
}


@dataclass
class PicoTypeConfig:
    max_bytes: int = 1024
    embed_dim: int = 96
    trunk_dim: int = 192
    num_heads: int = 4
    num_attn_layers: int = 2
    rope_theta: float = 500_000.0
    conv_kernels: Tuple[int, ...] = (3, 5, 7)
    tiers: Dict[str, int] = field(default_factory=lambda: dict(TIERS))
    num_coarse: int = 12
    num_modality: int = 8
    num_subtype: int = 24
    num_code_lang: int = 62
    num_text_lang: int = 30
    num_file_mime: int = 90
    num_risk: int = 6
    dropout: float = 0.1
    undetected_threshold: float = 0.4
    risk_threshold: float = 0.5
    pad_byte: int = 0

    def tier_dim(self, tier: str) -> int:
        if tier not in self.tiers:
            raise KeyError(f"unknown tier '{tier}', available: {list(self.tiers)}")
        return self.tiers[tier]


class ByteEmbed(nn.Module):
    def __init__(self, num_bytes: int = 256, dim: int = 96):
        super().__init__()
        self.embed = nn.Embedding(num_bytes, dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        if in_dim != out_dim:
            self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = self.conv(x)
        h = h.transpose(1, 2)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        h = h.transpose(1, 2)
        return h + residual


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class RotaryPosEmb(nn.Module):
    def __init__(self, dim: int, theta: float = 500_000.0, max_seq: int = 4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rope dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq = max_seq
        self._cached_seq = 0
        self._cached_dim = dim
        self._build_cache(max_seq, dim)

    def _build_cache(self, max_seq: int, dim: int) -> None:
        t = torch.arange(max_seq, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self._cached_seq = max_seq
        self._cached_dim = dim

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = q.size(-2)
        if seq > self._cached_seq:
            self._build_cache(max(seq, self._cached_seq * 2), self._cached_dim)
        cos = self.cos_cached[:, :, :seq, :]
        sin = self.sin_cached[:, :, :seq, :]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


class AttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_theta: float = 500_000.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryPosEmb(self.head_dim, rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        attn_mask_float = None
        if mask is not None:
            attn_mask_float = mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask_float = torch.where(attn_mask_float.to(torch.bool), 0.0, float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_float,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        x = x + self.resid_drop(self.out_proj(out))
        x = x + self.mlp(self.norm2(x))
        return x


class Pool(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            m = mask.unsqueeze(-1).to(x.dtype)
            denom = m.sum(dim=1).clamp(min=1.0)
            mean = (x * m).sum(dim=1) / denom
            x_for_max = x.masked_fill(m == 0, float("-inf"))
            max_ = x_for_max.max(dim=1).values
            x_for_std = x * m
            mean_for_std = x_for_std.sum(dim=1, keepdim=True) / denom.unsqueeze(1)
            sq = ((x_for_std - mean_for_std * m) ** 2) * m
            std = sq.sum(dim=1) / denom
            std = std.sqrt().clamp(min=0.0)
        else:
            mean = x.mean(dim=1)
            max_ = x.max(dim=1).values
            std = x.std(dim=1, unbiased=False)
        return torch.cat([mean, max_, std], dim=-1)


class MatryoshkaHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        tier_dims: Dict[str, int],
        out_dim: int,
    ):
        super().__init__()
        if in_dim != max(tier_dims.values()):
            raise ValueError(
                f"trunk dim {in_dim} must equal max tier dim {max(tier_dims.values())}"
            )
        self.tier_dims = dict(tier_dims)
        self.linears = nn.ModuleDict({
            name: nn.Linear(d, out_dim) for name, d in tier_dims.items()
        })
        for linear in self.linears.values():
            nn.init.zeros_(linear.bias)
            nn.init.normal_(linear.weight, std=0.02)

    def forward(self, x: torch.Tensor, tier: str) -> torch.Tensor:
        if tier not in self.linears:
            raise KeyError(f"unknown tier '{tier}', available: {list(self.linears)}")
        d = self.tier_dims[tier]
        sliced = x[..., :d]
        return self.linears[tier](sliced)


class PicoType(nn.Module):
    def __init__(self, config: Optional[PicoTypeConfig] = None):
        super().__init__()
        self.config = config or PicoTypeConfig()
        cfg = self.config

        self.embed = ByteEmbed(256, cfg.embed_dim)

        in_dim = cfg.embed_dim
        self.conv_blocks = nn.ModuleList()
        for k in cfg.conv_kernels:
            self.conv_blocks.append(
                ConvBlock(in_dim, cfg.trunk_dim, k, cfg.dropout)
            )
            in_dim = cfg.trunk_dim

        self.attn_blocks = nn.ModuleList([
            AttnBlock(cfg.trunk_dim, cfg.num_heads, cfg.dropout, cfg.rope_theta)
            for _ in range(cfg.num_attn_layers)
        ])

        self.pool = Pool()
        self.trunk_dim = 3 * cfg.trunk_dim

        self.heads = nn.ModuleDict({
            "coarse": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_coarse),
            "modality": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_modality),
            "subtype": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_subtype),
            "code_lang": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_code_lang),
            "text_lang": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_text_lang),
            "file_mime": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_file_mime),
            "risk": MatryoshkaHead(self.trunk_dim, cfg.tiers, cfg.num_risk),
        })

    def forward(
        self,
        bytes_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tier: str = "base",
    ) -> Dict[str, torch.Tensor]:
        if bytes_input.dtype != torch.long:
            bytes_input = bytes_input.long()
        if bytes_input.size(1) > self.config.max_bytes:
            bytes_input = bytes_input[:, : self.config.max_bytes]
            if mask is not None:
                mask = mask[:, : self.config.max_bytes]

        if mask is None:
            mask = torch.ones_like(bytes_input, dtype=torch.bool)
        else:
            mask = mask.to(device=bytes_input.device, dtype=torch.bool)
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True

        x = self.embed(bytes_input)
        x = x.transpose(1, 2)
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2)
        for block in self.attn_blocks:
            x = block(x, mask)
        pooled = self.pool(x, mask)
        return {name: head(pooled, tier) for name, head in self.heads.items()}

    def parameter_count(self, tier: Optional[str] = None) -> int:
        if tier is None:
            return sum(p.numel() for p in self.parameters())
        kept = 0
        for name, p in self.named_parameters():
            if ".linears." in name:
                if f".linears.{tier}." in name:
                    kept += p.numel()
            else:
                kept += p.numel()
        return kept

    def tier_sizes(self) -> Dict[str, int]:
        return {tier: self.parameter_count(tier) for tier in self.config.tiers}


def encode_bytes(data: bytes, max_len: int = 1024, pad: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = list(data[:max_len])
    seq = raw + [pad] * (max_len - len(raw))
    mask = [1] * len(raw) + [0] * (max_len - len(raw))
    return torch.tensor([seq], dtype=torch.long), torch.tensor([mask], dtype=torch.long)


def smoke_test() -> Dict[str, int]:
    cfg = PicoTypeConfig(max_bytes=128)
    model = PicoType(cfg).eval()
    sample = (
        b"def hello():\n    print('hi')\n\nif __name__ == '__main__':\n    hello()\n"
    )
    x, mask = encode_bytes(sample, max_len=cfg.max_bytes, pad=cfg.pad_byte)
    with torch.no_grad():
        logits = model(x, mask, tier="base")
    sizes = {tier: model.parameter_count(tier) for tier in cfg.tiers}
    return {
        "logits_keys": len(logits),
        "tiers": len(cfg.tiers),
        **sizes,
    }


if __name__ == "__main__":
    info = smoke_test()
    for k, v in info.items():
        print(f"{k}: {v}")
