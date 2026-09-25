"""Fit per-label Risk++ decision thresholds: LOWEST T with fit-spec >= floor.

Companion to arth_fit_temps.py (which calibrates semantic heads). Fit split:
gen seed 555xx + benign 88888. Eval split (report only): gen 666xx + benign 77777.
The audit (`data/arth_audit_external.json`) is NEVER fit on — eval only.

Policy history: the original Youden-J selector maximized J = sens + spec - 1.
With saturated in-distribution positives (sens ~1.0), J reduces to spec and is
therefore maximized by the HIGHEST T — hugging the hardest fit negative
(pii_card 0.999, sql 0.930). In-dist that looks perfect (recall/spec 1.00);
externally it is dead (OOD positives score 0.01-0.99 and never clear).
Replacement policy (2026-09-23): per-label specificity FLOOR on fit negatives,
take the LOWEST threshold satisfying it — in-dist recall is already saturated at
any such t, while OOD positives regain headroom. Floors: 0.97 default,
prompt_injection 0.90 (its near-miss benign family genuinely overlaps).

Writes scripts/risk_thresholds.json: {label: threshold}. Labels absent from the
file (jwt, ssh_key, password, email, phone) have NO trained positives anywhere
in training data — risk_flags defaults them to 0.5, and no threshold can fix
that (they need generators + a training round; see plan log 2026-09-23).

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


def fit_thresholds(pos: np.ndarray, neg: np.ndarray, min_spec: float = 0.97) -> float:
    """Lowest threshold t with (neg < t).mean() >= min_spec (risk_flags fires on
    >= t, so negatives strictly below t are correct rejections)."""
    if len(neg) == 0:
        return 0.5
    for v in np.unique(neg):
        if float((neg < v).mean()) >= min_spec:
            return float(v)
    return float(np.max(neg)) + 1e-4


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
    # api_key: positives ride secrets_aws + secrets_github + the dedicated
    # api_key generator (amendment #4); negatives likewise.
    from model.pico_type.arth_data import RISK14

    ak_idx = RISK14.index("api_key")
    pos, neg = [], []
    for gname in ("secrets_aws", "secrets_github", "api_key"):
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


# Per-label FP budgets: inject's near-miss benign family genuinely overlaps
# (fit-neg mass >=0.01 is 54%); a 0.97 floor would push its T above the shipped
# 0.47 and cut in-dist recall. Plan does not gate inject.
MIN_SPEC = {"prompt_injection": 0.90}
# Battery benign-probe floors (owner decision 2026-09-23, "balanced 3/11"):
# t must clear the channel scores of the two NEW benign-probe FP items so the
# probe list stays at 3/11 (2 pre-existing prompt_injection FPs accepted as the
# known inject limitation). Measured on ft3:
#   print('hello world')        sql=0.0307 xss=0.0260
#   SELECT id, name FROM users; jailbreak=0.3972 sql=0.3108 (sql flag ACCEPTED
#   for external sql_injection recall — residual FP = that one item)
# The audit is NEVER used as a floor source.
PROBE_FLOOR = {"sql_injection": 0.0311, "xss_payload": 0.0261, "jailbreak": 0.398}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_arth_ft3/arth_final.pt")
    args = ap.parse_args()
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    m = ArthModel(os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["student"])
    m.eval()

    fit = collect(m, 55500, 88888)
    thrs = {
        lab: round(max(fit_thresholds(pos, neg, min_spec=MIN_SPEC.get(lab, 0.97)),
                       PROBE_FLOOR.get(lab, 0.0)), 4)
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
