"""Fit per-(mode, count-bucket) temperatures on HELD-OUT synth (seed != 7).

Plan §2: "temps per (choice/score/noul x 2/3-5/6-20) on held-out. Ship
temperatures.json + refit.py. Report raw+fitted always." Gate: ECE fitted <=0.10.

Val data is regenerated from generator seeds 70001+ (train used 7) — unseen by
construction. Writes scripts/temperatures.json in Calibrator key format
"{mode}/{bucket}".

Usage:
  python scripts/arth_fit_temps.py                          # default ckpt
  python scripts/arth_fit_temps.py --ckpt checkpoints_arth/arth_step_3000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arth import ArthModel, bucket_count
from model.pico_type.arth_data import (
    make_choice_decisions,
    make_invoice_decisions,
    make_noul_decisions,
    make_score_decisions,
    make_unanswerable,
)

N_BINS = 10


def encode(batch: list[str], max_bytes: int = 1024):
    ids = [[b for b in s.encode("utf-8", errors="replace")[:max_bytes]] for s in batch]
    ln = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), ln, dtype=torch.long)
    mask = torch.zeros(len(ids), ln, dtype=torch.bool)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    return pad, mask


def forward_logits(m: ArthModel, items: list[dict], bs: int = 64) -> list[torch.Tensor]:
    """Semantic logits per item (K options, padded within batch)."""
    out: list[torch.Tensor] = []
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        jids, jmask = encode([s["input"] for s in chunk])
        pooled = m.pooled(jids, jmask)
        maxk = max(len(s["options"]) for s in chunk)
        bems, kmask = [], []
        for s in chunk:
            rows = []
            for o in s["options"]:
                ob = o.encode("utf-8", errors="replace")[:256] or b"\x00"
                rows.append(m.trunk.embed(torch.tensor([[b for b in ob]])).mean(dim=1))
            while len(rows) < maxk:
                rows.append(torch.zeros_like(rows[0]))
            bems.append(torch.cat(rows, dim=0))
            kmask.append([1] * len(s["options"]) + [0] * (maxk - len(s["options"])))
        emb = torch.stack(bems)
        km = torch.tensor(kmask, dtype=torch.bool)
        logits = m.semantic(pooled, emb)
        logits = logits.masked_fill(~km, float("nan"))
        for j, s in enumerate(chunk):
            out.append(logits[j, : len(s["options"])])
    return out


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, mode: str) -> float:
    """Scalar T minimizing NLL (choice CE) or BCE (score/noul) over log-grid + refine."""
    lo, hi = -3.0, 3.0  # T in [e^-3, e^3]

    def loss(log_t: float) -> float:
        t = float(np.exp(log_t))
        if mode == "choice":
            return float(F.cross_entropy(logits / t, targets))
        return float(F.binary_cross_entropy_with_logits(logits / t, targets))

    grid = np.linspace(lo, hi, 121)
    losses = [loss(x) for x in grid]
    best = float(grid[int(np.argmin(losses))])
    # local refine
    for scale in (0.1, 0.02, 0.005):
        cand = np.linspace(best - scale, best + scale, 21)
        ls = [loss(x) for x in cand]
        best = float(cand[int(np.argmin(ls))])
    return float(np.exp(best))


def ece_choice(logits: torch.Tensor, correct: torch.Tensor, t: float = 1.0) -> float:
    probs = F.softmax(logits / t, dim=-1)
    conf, pred = probs.max(dim=-1)
    right = (pred == correct).float()
    conf_np, right_np = conf.numpy(), right.numpy()
    bins = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0
    for b in range(N_BINS):
        sel = (conf_np > bins[b]) & (conf_np <= bins[b + 1])
        if sel.any():
            ece += abs(float(conf_np[sel].mean() - right_np[sel].mean())) * float(sel.mean())
    return ece


def ece_sigmoid(logits: torch.Tensor, targets: torch.Tensor, t: float = 1.0) -> float:
    p = torch.sigmoid(logits / t).numpy().ravel()
    y = targets.numpy().astype(float).ravel()
    bins = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0
    for b in range(N_BINS):
        sel = (p > bins[b]) & (p <= bins[b + 1])
        if sel.any():
            ece += abs(float(p[sel].mean() - y[sel].mean())) * float(sel.mean())
    return ece


def build_val() -> dict[tuple[str, str], tuple[list[dict], str]]:
    """(mode, bucket) -> (items, loss_kind) from HELD-OUT seeds (train used 7)."""
    pools: list[tuple[str, list[dict]]] = [
        ("choice", make_choice_decisions(800, 70001)),
        ("choice", make_invoice_decisions(300, 70004)),
        ("score", make_score_decisions(300, 70002)),
        ("noul", make_noul_decisions(300, 70003)),
        ("noul", make_unanswerable(300, 70005)),
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for mode, items in pools:
        for it in items:
            key = (it["mode"], bucket_count(len(it["options"])))
            grouped.setdefault(key, []).append(it)
    kind = {"choice": "ce", "score": "bce", "noul": "bce"}
    return {k: (v, kind[k[0]]) for k, v in grouped.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_arth_ft3/arth_final.pt")
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "temperatures.json"))
    args = ap.parse_args()

    ckpt = args.ckpt
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(ROOT, ckpt)
    if not os.path.exists(ckpt):
        sys.exit(f"ckpt not found: {ckpt}")

    m = ArthModel(os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["student"])
    m.eval()

    temps: dict[str, float] = {}
    print(f"{'group':<16} {'n':>5} {'rawECE':>8} {'T':>6} {'fitECE':>8}")
    with torch.no_grad():
        for (mode, bucket), (items, kind) in sorted(build_val().items()):
            logits_l = forward_logits(m, items)
            if kind == "ce":
                lg = torch.stack(logits_l)
                tgt = torch.tensor([it["correct"] for it in items], dtype=torch.long)
                raw = ece_choice(lg, tgt)
                t = fit_temperature(lg, tgt, "choice")
                fit = ece_choice(lg, tgt, t)
            else:
                lg = torch.stack([x.nan_to_num(-1e9) if x.isnan().any() else x for x in logits_l])
                tgt = torch.stack(
                    [
                        torch.tensor(
                            it["scores"] if mode == "score" else [0] * len(it["options"]),
                            dtype=torch.float,
                        )
                        for it in items
                    ]
                )
                raw = ece_sigmoid(lg, tgt)
                t = fit_temperature(lg, tgt, mode)
                fit = ece_sigmoid(lg, tgt, t)
            key = f"{mode}/{bucket}"
            temps[key] = round(t, 4)
            print(f"{key:<16} {len(items):>5} {raw:>8.4f} {t:>6.3f} {fit:>8.4f}")

    with open(args.out, "w") as f:
        json.dump(temps, f, indent=2)
    print(f"wrote {args.out}")
    worst = max(temps, key=lambda k: 0)
    print("note: raw vs fitted ECE above; gate fitted <= 0.10 (plan Wk3-4)")
    del worst


if __name__ == "__main__":
    main()
