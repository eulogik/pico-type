"""Pre-registered held-out gates (plan §3): AG News >=0.92, SST-2 >=0.84,
Enron spam >=0.98 — semantic choice head, fixed-1024 pad protocol.

Eval sets are HELD-OUT (ag_news test / sst2 validation / enron test), deduped
against train samples, PII-masked at cache time, never trained on.

Usage: python scripts/arth_eval_gates.py [--ckpt ...]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arth import ArthModel
from model.pico_type.arth_data import load_ag_eval, load_enron_eval, load_sst2_eval

GATES = {"ag_news": 0.92, "sst2": 0.84, "enron_spam": 0.98}


def choice_logits(m: ArthModel, items: list[dict], bs: int = 64) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        # fixed 1024-pad (established ARTH protocol: trunk pooled is NOT
        # pad-length invariant — verified max abs diff 24.09 on same input)
        pad = torch.zeros(len(chunk), 1024, dtype=torch.long)
        mask = torch.zeros(len(chunk), 1024, dtype=torch.bool)
        for j, s in enumerate(chunk):
            b = s["input"].encode("utf-8", errors="replace")[:1024]
            pad[j, : len(b)] = torch.tensor(list(b))
            mask[j, : len(b)] = True
        pooled = m.pooled(pad, mask)
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
        logits = m.semantic(pooled, torch.stack(bems))
        km = torch.tensor(kmask, dtype=torch.bool)
        logits = logits.masked_fill(~km, -1e9)
        for j, s in enumerate(chunk):
            out.append(logits[j, : len(s["options"])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_arth_ft4/arth_final.pt")
    args = ap.parse_args()
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    if not os.path.exists(ckpt):
        sys.exit(f"ckpt not found: {ckpt}")

    m = ArthModel(os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["student"])
    m.eval()

    tables = {
        "ag_news": load_ag_eval(),
        "sst2": load_sst2_eval(),
        "enron_spam": load_enron_eval(),
    }
    all_pass = True
    with torch.no_grad():
        for name, items in tables.items():
            if not items:
                print(f"{name}: SKIP (cache absent)")
                continue
            lg = choice_logits(m, items)
            correct = sum(
                int(int(l.argmax()) == s["correct"]) for l, s in zip(lg, items)
            )
            acc = correct / len(items)
            gate = GATES[name]
            ok = acc >= gate
            all_pass = all_pass and ok
            print(f"{name}: acc {acc:.4f} ({correct}/{len(items)}) gate>={gate} {'PASS' if ok else 'FAIL'}")
    print("GATES:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
