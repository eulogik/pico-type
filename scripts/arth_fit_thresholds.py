"""Fit per-label Risk++ decision thresholds on held-out data (Youden's J).

Companion to arth_fit_temps.py (which calibrates semantic heads). The risk head
ranks well (AUC 0.95-1.0) but a flat 0.5 threshold is the wrong operating point
(FP audit: benign false positives at 0.5). Fit split: gen seed 555xx + benign
88888. Eval split (report only): gen 666xx + benign 77777.

Writes scripts/risk_thresholds.json: {label: threshold}. Defaults to 0.5 when
a label has no signal.

Usage: python scripts/arth_fit_thresholds.py [--ckpt ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arth import RISK_PLUS_LABELS, ArthModel
from model.pico_type.arth_data import RISKPP_GENERATORS, gen_benign_hard


def scores_for(m: ArthModel, items: list[dict], label: str) -> list[float]:
    idx = RISK_PLUS_LABELS.index(label)
    out = []
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([[b for b in it["input"].encode("utf-8", errors="replace")[:1024]]])
            mask = torch.ones_like(ids, dtype=torch.bool)
            out.append(float(torch.sigmoid(m.riskpp(m.pooled(ids, mask)))[0][idx]))
    return out


def fit_thresholds(pos: np.ndarray, neg: np.ndarray, min_spec: float = 0.97, eps: float = 0.005) -> float:
    """Among thresholds with specificity >= min_spec and J within eps of the best
    achievable J, return the LOWEST T. The eps band keeps near-optimal operating
    points from hugging the fit-min-positive (brittle to OOD demo strings that
    score just below fit positives but above almost all negatives)."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    cand = np.unique(np.concatenate([pos, neg]))
    scored = []
    fallback = (None, -1.0, 0.5)
    for t in cand:
        sens = float((pos >= t).mean())
        spec = float((neg < t).mean())
        j = sens + spec - 1.0
        if spec >= min_spec:
            scored.append((float(t), j))
        if j > fallback[1]:
            fallback = (spec, j, float(t))
    if not scored:
        return fallback[2]
    best_j = max(j for _, j in scored)
    eligible = [t for t, j in scored if j >= best_j - eps]
    if not eligible:
        return fallback[2]
    return min(eligible)


def collect(m: ArthModel, gen_seed_base: int, benign_seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    benign = gen_benign_hard(400, benign_seed)
    result = {}
    for gi, (gname, gen) in enumerate(RISKPP_GENERATORS.items()):
        pos, neg = [], []
        for it in gen(80, gen_seed_base + gi):
            s = scores_for(m, [it], gname)[0]
            (pos if any(it["risk14"]) else neg).append(s)
        neg.extend(scores_for(m, benign, gname))
        result[gname] = (np.array(pos), np.array(neg))
    # api_key: labeled by secrets_aws + secrets_github positives (RISK14 index 4 is email;
    # api_key is index 0 in RISK14)
    from model.pico_type.arth_data import RISK14

    ak_idx = RISK14.index("api_key")
    pos, neg = [], []
    for gi, gname in enumerate(("secrets_aws", "secrets_github")):
        gen = RISKPP_GENERATORS[gname]
        for it in gen(80, gen_seed_base + list(RISKPP_GENERATORS).index(gname)):
            s = scores_for(m, [it], "api_key")[0]
            (pos if it["risk14"][ak_idx] else neg).append(s)
    neg.extend(scores_for(m, benign, "api_key"))
    result["api_key"] = (np.array(pos), np.array(neg))
    return result


def report(name: str, data: dict[str, tuple[np.ndarray, np.ndarray]], thrs: dict[str, float]) -> None:
    print(f"-- {name}")
    for lab, (pos, neg) in data.items():
        t = thrs.get(lab, 0.5)
        sens = float((pos >= t).mean()) if len(pos) else float("nan")
        spec = float((neg < t).mean()) if len(neg) else float("nan")
        auc = float("nan")
        if len(pos) and len(neg):
            s = np.concatenate([pos, neg])
            y = np.array([1] * len(pos) + [0] * len(neg))
            order = np.argsort(s)
            ranks = np.empty(len(s))
            ranks[order] = np.arange(1, len(s) + 1)
            n_p, n_n = int(y.sum()), int((1 - y).sum())
            auc = (ranks[y == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)
        print(f"  {lab:<18} T={t:.3f} recall={sens:.3f} spec={spec:.3f} auc={auc:.3f}")


# Per-label FP budgets: inject's ROC genuinely overlaps (AUC~0.96) — a tight
# spec budget collapses recall (<0.5); plan does not gate inject. Others tight.
MIN_SPEC = {"prompt_injection": 0.90}
# eps-floor ONLY for demo-critical secret labels: their OOD demo strings (bare
# AKIA...EXAMPLE, ghp_EXAMPLE) score just below fit-min-positive, so a strict
# max-J T (hugging fit min-pos) misses them. Floor admits ~0.5% fit-neg slack.
# All other labels stay strict max-J (flooring them caused benign FPs).
FLOOR_EPS = {"api_key": 0.005, "secrets_aws": 0.005, "secrets_github": 0.005}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_arth_ft/arth_final.pt")
    args = ap.parse_args()
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    m = ArthModel(os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["student"])
    m.eval()

    fit = collect(m, 55500, 88888)
    thrs = {
        lab: round(
            fit_thresholds(pos, neg, min_spec=MIN_SPEC.get(lab, 0.97),
                           eps=FLOOR_EPS.get(lab, 0.0)),
            4,
        )
        for lab, (pos, neg) in fit.items()
    }
    ev = collect(m, 66600, 77777)
    report("fit (thresholds chosen here)", fit, thrs)
    report("eval (unseen seeds)", ev, thrs)

    out = os.path.join(ROOT, "scripts", "risk_thresholds.json")
    with open(out, "w") as f:
        json.dump(thrs, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
