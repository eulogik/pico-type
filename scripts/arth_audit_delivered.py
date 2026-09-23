"""External hand-labeled delivered-error report (plan §3 Cheap-Verifiers rule).

arXiv:2609.01345: a guardrail that grades itself is blind by construction —
in-loop verifier dashboards read 3% error while true error hits 32%. Every
delivered-error number for Risk++/act MUST come from `data/arth_audit_external.json`
(hand-labeled, never trained on, never used for threshold/temperature fitting).

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
    delivered_error = (total_fn + total_fp) / max(1, total_flags)

    print(f"external hand-labeled audit: {n} items x {len(RISK_PLUS_LABELS)} labels = {total_flags} decisions")
    print(f"{'label':<18} {'pos':>4} {'neg':>4} {'FN':>3} {'FP':>3}  recall  spec")
    for lbl in RISK_PLUS_LABELS:
        v = per_label[lbl]
        rec = (v["pos"] - v["fn"]) / max(1, v["pos"])
        spec = (v["neg"] - v["fp"]) / max(1, v["neg"])
        mark = " *" if (v["fn"] or v["fp"]) and (v["pos"] or v["neg"] > 5) else ""
        print(f"{lbl:<18} {v['pos']:>4} {v['neg']:>4} {v['fn']:>3} {v['fp']:>3}  {rec:>6.3f} {spec:>6.3f}{mark}")
    print(f"\nitem exact-match: {item_exact}/{n} = {item_exact/n:.3f}")
    print(f"DELIVERED ERROR (flag-level, external labels): {delivered_error:.4f}")
    print("source: data/arth_audit_external.json (hand-labeled; never trained/fit)")
    print("protocol: arXiv:2609.01345 — never recompute this through the model itself")


if __name__ == "__main__":
    main()
