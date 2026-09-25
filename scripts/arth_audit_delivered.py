"""External hand-labeled delivered-error report (plan §3 Cheap-Verifiers rule).

arXiv:2609.01345: a guardrail that grades itself is blind by construction —
in-loop verifier dashboards read 3% error while true error hits 32%. Every
delivered-error number for Risk++/act MUST come from `data/arth_audit_external.json`
(hand-labeled, never trained on, never used for threshold/temperature fitting).

Metric note (2026-09-23): flag-level error over the 14-label vector is
negatively dominated (2842 neg vs 112 pos) and therefore rewards not flagging —
an all-negative predictor scores lower than a working guardrail. The PRIMARY
delivered metric is positive recall (+ balanced error); flag-level error is
printed only as a secondary number next to its all-negative baseline.

Usage: python scripts/arth_audit_delivered.py [--ckpt ...]
Exit code: 0 = report written (informational; no numeric gate is pre-registered
for the audit — numbers are for release/paper citation only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arth import RISK_PLUS_LABELS, ArthModel, load_risk_thresholds, risk_flags

AUDIT_PATH = os.path.join(ROOT, "data", "arth_audit_external.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_arth_ft3/arth_final.pt")
    args = ap.parse_args()
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)

    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    items = audit["items"]
    thrs = load_risk_thresholds()

    m = ArthModel(os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["student"])
    m.eval()

    per_label = {lbl: {"fn": 0, "fp": 0, "pos": 0, "neg": 0} for lbl in RISK_PLUS_LABELS}
    item_exact = 0
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([[b for b in it["text"].encode("utf-8", errors="replace")[:1024]]])
            mask = torch.ones_like(ids, dtype=torch.bool)
            p = torch.sigmoid(m.riskpp(m.pooled(ids, mask)))
            flags = risk_flags(p, thrs)[0]
            want = torch.tensor(it["risk14"], dtype=torch.bool)
            if bool((flags == want).all()):
                item_exact += 1
            for i, lbl in enumerate(RISK_PLUS_LABELS):
                if want[i]:
                    per_label[lbl]["pos"] += 1
                    if not bool(flags[i]):
                        per_label[lbl]["fn"] += 1
                else:
                    per_label[lbl]["neg"] += 1
                    if bool(flags[i]):
                        per_label[lbl]["fp"] += 1

    n = len(items)
    total_fn = sum(v["fn"] for v in per_label.values())
    total_fp = sum(v["fp"] for v in per_label.values())
    total_flags = sum(v["pos"] + v["neg"] for v in per_label.values())
    total_pos = sum(v["pos"] for v in per_label.values())
    total_neg = total_flags - total_pos
    delivered_error = (total_fn + total_fp) / max(1, total_flags)
    pos_recall = (total_pos - total_fn) / max(1, total_pos)
    neg_spec = (total_neg - total_fp) / max(1, total_neg)
    allneg_error = total_pos / max(1, total_flags)

    print(f"external hand-labeled audit: {n} items x {len(RISK_PLUS_LABELS)} labels = {total_flags} decisions")
    print(f"{'label':<18} {'pos':>4} {'neg':>4} {'FN':>3} {'FP':>3}  recall  spec")
    for lbl in RISK_PLUS_LABELS:
        v = per_label[lbl]
        rec = (v["pos"] - v["fn"]) / max(1, v["pos"])
        spec = (v["neg"] - v["fp"]) / max(1, v["neg"])
        mark = " *" if (v["fn"] or v["fp"]) and (v["pos"] or v["neg"] > 5) else ""
        print(f"{lbl:<18} {v['pos']:>4} {v['neg']:>4} {v['fn']:>3} {v['fp']:>3}  {rec:>6.3f} {spec:>6.3f}{mark}")
    print(f"\nitem exact-match: {item_exact}/{n} = {item_exact/n:.3f}")
    print(f"POSITIVE RECALL (primary delivered metric): {total_pos - total_fn}/{total_pos} = {pos_recall:.4f}")
    print(f"negative specificity: {total_neg - total_fp}/{total_neg} = {neg_spec:.4f}")
    print(f"balanced error 0.5*(FNR+FPR): {0.5*((1 - pos_recall) + (1 - neg_spec)):.4f}")
    print(f"flag-level error: {delivered_error:.4f}  ** SECONDARY — {total_neg}:{total_pos} neg:pos "
          f"dominance; all-negative predictor scores {allneg_error:.4f}, i.e. flag-level error "
          f"REWARDS not flagging (the Cheap-Verifiers trap, arXiv:2609.01345). Cite positive "
          f"recall / balanced error, never flag-level error alone.")
    print("source: data/arth_audit_external.json (hand-labeled; never trained/fit)")
    print("protocol: arXiv:2609.01345 — never recompute this through the model itself")


if __name__ == "__main__":
    main()
