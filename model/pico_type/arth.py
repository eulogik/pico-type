"""ARTH (V2) heads: semantic decider, relational scorer, act/escalate, calibrator.

Wk 1-2 status: wiring + stubs. All new heads are randomly initialized and
UNTRAINED — they run end-to-end (shapes, demos, latency) but carry no signal
until Wk 3-4 training. The trunk is the frozen recovered v0.2 trunk
(checkpoints/arth_trunk_v02.pt); legacy heads are untouched (parity gate).

Design (per plan §1):
- options are marked by BYTE OFFSET spans (start, end) — no token ids.
- semantic scorer: [ctx_192, opt_192...] -> 2-layer pre-norm Transformer (192d)
  -> 1 logit/option -> choice softmax | score sigmoid | noul sigmoid.
- relational scorer: deterministic structural byte features + linear head.
- calibrator: per (mode, count-bucket) temperatures, default 1.0.
- act head: [pooled_576, top1, margin, entropy, k/255] -> act/escalate logits.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .arch import TIERS, MatryoshkaHead, PicoType, PicoTypeConfig
from .labels import RISK_LABELS

RISK_PLUS_NEW = [
    "prompt_injection",
    "jailbreak",
    "pii_ssn",
    "pii_card",
    "secrets_aws",
    "secrets_github",
    "sql_injection",
    "xss_payload",
]
RISK_PLUS_LABELS = list(RISK_LABELS) + RISK_PLUS_NEW
assert len(RISK_PLUS_LABELS) == 14

SEP = b"\x00"
MAX_INPUT_BYTES = 1536
MAX_OPTION_BYTES = 256
COUNT_BUCKETS = ((2, 2), (3, 5), (6, 20))


def bucket_count(k: int) -> str:
    for lo, hi in COUNT_BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return "6-20" if k > 20 else "2-2"


def mark_options(data: bytes, options: list[bytes]) -> tuple[bytes, list[tuple[int, int]]]:
    """Join input + NUL-separated options; return bytes + per-option (start, end) byte offsets."""
    inp = data[:MAX_INPUT_BYTES]
    spans: list[tuple[int, int]] = []
    parts = [inp]
    for opt in options:
        parts.append(SEP)
        start = sum(len(p) for p in parts)
        ob = opt[:MAX_OPTION_BYTES]
        parts.append(ob)
        spans.append((start, start + len(ob)))
    return b"".join(parts), spans


def structural_features(data: bytes) -> np.ndarray:
    """Deterministic byte-level structural features (no learned params).

    Captures bracket balance, delimiter density, line structure — the raw
    material a relational head reasons over (AST patterns, field layouts).
    """
    b = data[:4096]
    n = max(1, len(b))
    pairs = [(ord("{"), ord("}")), (ord("["), ord("]")), (ord("("), ord(")")), (ord("<"), ord(">"))]
    feats: list[float] = []
    depth = 0
    max_depth = 0
    imbalance = 0
    for byte in b:
        for o, c in pairs:
            if byte == o:
                depth += 1
                max_depth = max(max_depth, depth)
            elif byte == c:
                if depth > 0:
                    depth -= 1
                else:
                    imbalance += 1
    feats += [max_depth / 32.0, imbalance / 32.0, depth / 32.0]
    for ch in b";:,=\\\"'":
        feats.append(b.count(ch) / n)
    lines = b.split(b"\n")
    feats.append(len(lines) / 64.0)
    if lines:
        lens = [len(x) for x in lines]
        feats.append(sum(lens) / (len(lens) * 128.0))
        feats.append((max(lens) if lens else 0) / 512.0)
    else:
        feats += [0.0, 0.0]
    feats.append(sum(1 for x in lines if x[:1] in (b" ", b"\t")) / max(1, len(lines)))
    return np.array(feats, dtype=np.float32)


N_STRUCT_FEATS = len(structural_features(b""))


class PreNormTransformer(nn.Module):
    def __init__(self, dim: int = 192, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "n1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                        "n2": nn.LayerNorm(dim),
                        "ff1": nn.Linear(dim, 4 * dim),
                        "ff2": nn.Linear(4 * dim, dim),
                        "drop": nn.Dropout(dropout),
                    }
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            h, _ = lyr["attn"](lyr["n1"](x), lyr["n1"](x), lyr["n1"](x), need_weights=False)
            x = x + lyr["drop"](h)
            x = x + lyr["drop"](lyr["ff2"](F.gelu(lyr["ff1"](lyr["n2"](x)))))
        return x


class SemanticHead(nn.Module):
    """Per-option scorer. UNTRAINED stub in Wk 1-2 (random init, runs + shapes only)."""

    def __init__(self, trunk_dim: int = 576, embed_dim: int = 96, dim: int = 192, heads: int = 4):
        super().__init__()
        self.ctx_proj = nn.Linear(trunk_dim, dim)
        self.opt_proj = nn.Linear(embed_dim, dim)
        self.encoder = PreNormTransformer(dim, heads, 2)
        self.scorer = nn.Linear(dim, 1)

    def forward(self, pooled: torch.Tensor, opt_embs: torch.Tensor) -> torch.Tensor:
        """pooled: (B,576) trunk output; opt_embs: (B,K,96) mean byte-embeds per option."""
        ctx = self.ctx_proj(pooled).unsqueeze(1)
        opts = self.opt_proj(opt_embs)
        seq = torch.cat([ctx, opts], dim=1)
        enc = self.encoder(seq)
        return self.scorer(enc[:, 1:, :]).squeeze(-1)


class RelationalHead(nn.Module):
    """Structural scorer over [pooled, structural feats]. UNTRAINED stub in Wk 1-2."""

    def __init__(self, trunk_dim: int = 576, n_feats: int = N_STRUCT_FEATS):
        super().__init__()
        self.scorer = nn.Linear(trunk_dim + n_feats, 1)

    def forward(self, pooled: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return self.scorer(torch.cat([pooled, feats], dim=1)).squeeze(-1)


class RiskPlusPlusHead(nn.Module):
    """14-label multi-label risk head (Matryoshka slices). UNTRAINED stub in Wk 1-2,
    except the 6 overlapping labels warm-started from the frozen v0.2 risk head."""

    def __init__(self, trunk_dim: int = 576, tiers: dict | None = None):
        super().__init__()
        self.head = MatryoshkaHead(trunk_dim, dict(tiers or TIERS), len(RISK_PLUS_LABELS))

    def warm_start_from(self, trunk_risk) -> None:
        """Copy the 6 learned v0.2 risk rows per tier; 8 new rows stay random-init."""
        with torch.no_grad():
            for tier, lin in self.head.linears.items():
                src = trunk_risk.linears[tier]
                lin.weight[: len(RISK_LABELS)].copy_(src.weight)
                lin.bias[: len(RISK_LABELS)].copy_(src.bias)

    def forward(self, pooled: torch.Tensor, tier: str = "base") -> torch.Tensor:
        return self.head(pooled, tier)


class ActHead(nn.Module):
    """Act vs escalate from [pooled, top1, margin, entropy, k/255]. UNTRAINED stub."""

    def __init__(self, trunk_dim: int = 576):
        super().__init__()
        self.fc = nn.Linear(trunk_dim + 4, 2)

    def forward(self, pooled: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([pooled, stats], dim=1))


class Calibrator:
    """Per (mode, count-bucket) temperatures. Defaults 1.0 until Wk 3-4 refit."""

    def __init__(self, path: str | None = None):
        self.temps: dict[str, float] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self.temps = json.load(f)

    def temp(self, mode: str, k: int) -> float:
        return self.temps.get(f"{mode}/{bucket_count(k)}", 1.0)

    def apply(self, logits: torch.Tensor, mode: str) -> torch.Tensor:
        t = self.temp(mode, logits.shape[-1])
        if mode == "choice":
            return torch.softmax(logits / t, dim=-1)
        return torch.sigmoid(logits / t)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.temps, f, indent=2)


def dist_stats(probs: torch.Tensor) -> torch.Tensor:
    """[top1, margin, entropy, k/255] per row."""
    top2 = torch.topk(probs, min(2, probs.shape[-1])).values
    top1 = top2[:, 0]
    margin = top1 - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top1))
    entropy = -(probs * (probs + 1e-12).log()).sum(-1)
    k = torch.full_like(top1, probs.shape[-1] / 255.0)
    return torch.stack([top1, margin, entropy, k], dim=-1)


class ArthModel(nn.Module):
    """Frozen v0.2 trunk + new (untrained) V2 heads."""

    def __init__(self, trunk_ckpt: str, max_bytes: int = 1024):
        super().__init__()
        self.trunk = PicoType(PicoTypeConfig(max_bytes=max_bytes))
        sd = torch.load(trunk_ckpt, map_location="cpu", weights_only=False)
        self.trunk.load_state_dict(sd.get("model_state_dict", sd))
        self.trunk.requires_grad_(False)
        self.trunk.eval()
        self.semantic = SemanticHead()
        self.relational = RelationalHead()
        self.riskpp = RiskPlusPlusHead()
        self.riskpp.warm_start_from(self.trunk.heads["risk"])
        self.act = ActHead()
        self.calibrator = Calibrator()

    def pooled(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.trunk.embed(ids).transpose(1, 2)
            for b in self.trunk.conv_blocks:
                x = b(x)
            x = x.transpose(1, 2)
            for b in self.trunk.attn_blocks:
                x = b(x, mask)
            return self.trunk.pool(x, mask)

    def legacy(self, ids: torch.Tensor, mask: torch.Tensor, tier: str = "base") -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.trunk(ids, mask, tier=tier)

    def option_embeds(self, joined: bytes, spans: list[tuple[int, int]]) -> torch.Tensor:
        """Mean frozen byte-embeds per option span -> (K,96)."""
        with torch.no_grad():
            out = []
            for s, e in spans:
                ob = joined[s:e] or b"\x00"
                ids = torch.tensor([[b for b in ob]], dtype=torch.long)
                out.append(self.trunk.embed(ids).mean(dim=1))
            return torch.cat(out, dim=0)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


def count_params(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())
