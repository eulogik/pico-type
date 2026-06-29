"""pico-type: ONNX export and quantization pipeline."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .arch import PicoType, PicoTypeConfig, TIERS
from .labels import ALL_HEADS, HEAD_NUM_CLASSES


def _tier_wrapper(model: PicoType, tier: str) -> nn.Module:
    """Build a tier-specific wrapper with ONNX-compatible manual attention."""

    class AttnBlockONNX(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.num_heads = src.num_heads
            self.head_dim = src.head_dim
            self.qkv = src.qkv
            self.out_proj = src.out_proj
            self.norm1 = src.norm1
            self.norm2 = src.norm2
            self.mlp = src.mlp
            self.rope = src.rope

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = self.rope(q, k)
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask_float = mask[:, None, None, :].to(dtype=x.dtype)
            masked_value = torch.full_like(mask_float, -1e4)
            mask_float = torch.where(mask_float.to(torch.bool), torch.zeros_like(mask_float), masked_value)
            scores = scores + mask_float
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, D)
            x = x + self.out_proj(out)
            x = x + self.mlp(self.norm2(x))
            return x

    class PicoTypeTier(nn.Module):
        def __init__(self, src: PicoType, tier: str):
            super().__init__()
            self.embed = src.embed
            self.conv_blocks = src.conv_blocks
            self.attn_blocks = nn.ModuleList([AttnBlockONNX(b) for b in src.attn_blocks])
            self.pool = src.pool
            d = src.config.tier_dim(tier)
            self.heads = nn.ModuleDict({
                name: head.linears[tier] for name, head in src.heads.items()
            })
            self.tier_dim = d

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
            x = self.embed(input_ids.long())
            x = x.transpose(1, 2)
            for b in self.conv_blocks:
                x = b(x)
            x = x.transpose(1, 2)
            mask = attention_mask.to(torch.bool)
            for b in self.attn_blocks:
                x = b(x, mask)
            pooled = self.pool(x, mask)
            sliced = pooled[..., : self.tier_dim]
            return {name: mod(sliced) for name, mod in self.heads.items()}

    return PicoTypeTier(model, tier)


def export_to_onnx(
    model: PicoType,
    output_dir: str,
    tier: str = "base",
    opset: int = 18,
    max_bytes: Optional[int] = None,
) -> str:
    model.eval()
    cfg = model.config
    max_len = max_bytes or cfg.max_bytes

    wrapper = _tier_wrapper(model, tier).eval()
    dummy_ids = torch.zeros(1, max_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.bool)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"picotype_{tier}.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        path,
        opset_version=opset,
        external_data=False,
        input_names=["input_ids", "attention_mask"],
        output_names=list(ALL_HEADS),
        dynamic_shapes=(
            {0: "batch_size", 1: "sequence_length"},
            {0: "batch_size", 1: "sequence_length"},
        ),
    )
    return path


def export_all_tiers(
    model: PicoType,
    output_dir: str,
    opset: int = 18,
    max_bytes: Optional[int] = None,
) -> Dict[str, str]:
    paths = {}
    for tier in TIERS:
        p = export_to_onnx(model, output_dir, tier, opset, max_bytes)
        paths[tier] = p
    meta = {
        "tiers": list(TIERS),
        "tier_dims": TIERS,
        "params": {t: model.parameter_count(t) for t in TIERS},
        "head_classes": dict(HEAD_NUM_CLASSES),
        "max_bytes": max_bytes or model.config.max_bytes,
        "opset": opset,
    }
    with open(os.path.join(output_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return paths


def quantize_dynamic(
    onnx_path: str,
    output_path: Optional[str] = None,
) -> str:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError("install onnx and onnxruntime: pip install onnx onnxruntime")

    out = output_path or onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(onnx_path, out, weight_type=QuantType.QInt8)
    return out


def export_with_quantization(
    model: PicoType,
    output_dir: str,
    tier: str = "base",
    opset: int = 18,
    max_bytes: Optional[int] = None,
) -> Dict[str, str]:
    fp32_path = export_to_onnx(model, output_dir, tier, opset, max_bytes)
    int8_path = quantize_dynamic(fp32_path)

    sizes = {
        "fp32_bytes": os.path.getsize(fp32_path),
        "int8_bytes": os.path.getsize(int8_path),
    }
    with open(os.path.join(output_dir, f"picotype_{tier}_sizes.json"), "w") as f:
        json.dump(sizes, f, indent=2)
    return {"fp32": fp32_path, "int8": int8_path}


def smoke_export(max_len: int = 128) -> Dict[str, int]:
    import tempfile
    cfg = PicoTypeConfig(max_bytes=max_len)
    model = PicoType(cfg).eval()
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_all_tiers(model, tmp, opset=18, max_bytes=max_len)
        sizes = {}
        for tier, p in paths.items():
            sizes[tier] = os.path.getsize(p)
        return sizes


if __name__ == "__main__":
    sizes = smoke_export()
    for tier, sz in sizes.items():
        print(f"  {tier:5s}: {sz/1024:.1f} KB")
