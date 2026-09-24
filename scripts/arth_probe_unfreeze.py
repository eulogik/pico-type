"""Wk5+ A/B probe: can the FROZEN trunk form AG-topic features? (plan amendment queue (a))

The frozen-trunk ceiling probe (2026-09-23) held AG at 0.356 across epochs 3-8
with the trunk frozen, train loss still 1.2-1.5 — representation-bound. This
repeats the EXACT probe (ag_news 4000 train, 1200 steps x batch 32, held-out
eval every 200, seed 0 sampling) with the TRUNK UNFROZEN (lr 1e-4) so the only
changed variable is trunk plasticity. Frozen reference curve: 0.282 -> 0.294 ->
0.354 -> 0.352 -> 0.352 -> 0.356 (it0..it1000).

Research artifact only: nothing here touches the shipped base tier or its
parity baseline; no checkpoint is written unless --save is passed.

Caveat on the 2026-09-23 run recorded in the plan log: the first execution
evaluated in train mode (norm layers), which made repeated evals of identical
weights differ (0.634 vs 0.660) and inflated noise. Both A/B arms shared the
protocol, so the frozen-vs-unfrozen comparison stands; absolute values are
approximate. acc() now switches eval/train mode properly.

Usage:
  python scripts/arth_probe_unfreeze.py [--steps 1200] [--lr-trunk 1e-4] [--save]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arth import ArthModel
from model.pico_type.arth_data import gen_ag_news, load_ag_eval

FT3 = os.path.join(ROOT, "checkpoints_arth_ft3", "arth_final.pt")
TRUNK = os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt")


def fwd(m: ArthModel, items: list[dict]) -> torch.Tensor:
    pad = torch.zeros(len(items), 1024, dtype=torch.long)
    mask = torch.zeros(len(items), 1024, dtype=torch.bool)
    for j, s in enumerate(items):
        b = s["input"].encode("utf-8", errors="replace")[:1024]
        if b:
            pad[j, : len(b)] = torch.tensor(list(b))
            mask[j, : len(b)] = True
    pooled = m.pooled(pad, mask)
    maxk = max(len(s["options"]) for s in items)
    bems, kmask = [], []
    for s in items:
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
    return logits.masked_fill(~km, -1e9)


@torch.no_grad()
def acc(m: ArthModel, items: list[dict], bs: int = 128) -> float:
    was_training = m.training
    m.eval()
    hits = []
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        lg = fwd(m, chunk)
        hits.append((lg.argmax(-1) == torch.tensor([s["correct"] for s in chunk])).float())
    m.train(was_training)
    return float(torch.cat(hits).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-trunk", type=float, default=1e-4)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    m = ArthModel(TRUNK)
    m.load_state_dict(torch.load(FT3, map_location="cpu", weights_only=False)["student"])
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    heads = [p for n, p in m.named_parameters() if not n.startswith("trunk.")]
    trunk = [p for n, p in m.named_parameters() if n.startswith("trunk.")]
    opt = torch.optim.AdamW(
        [{"params": heads, "lr": args.lr, "weight_decay": 0.01},
         {"params": trunk, "lr": args.lr_trunk, "weight_decay": 0.01}]
    )
    trunk_before = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith("trunk.")}

    train = gen_ag_news(4000)
    ev = load_ag_eval(500)
    print(f"n_train={len(train)} n_eval={len(ev)} steps={args.bs and args.steps} lr_trunk={args.lr_trunk}", flush=True)
    print(f"step0 heldout: {acc(m, ev):.4f}  (frozen-trunk reference: 0.306)", flush=True)

    rng = random.Random(0)
    for it in range(args.steps):
        batch = rng.sample(train, args.bs)
        opt.zero_grad()
        lg = fwd(m, batch)
        loss = F.cross_entropy(lg, torch.tensor([s["correct"] for s in batch]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if it % 200 == 0 or it == args.steps - 1:
            print(f"it{it} loss={float(loss.detach()):.3f} heldout={acc(m, ev):.4f}", flush=True)

    train_acc = acc(m, train[:1000])
    delta = max(
        float((p.detach() - trunk_before[n]).abs().max())
        for n, p in m.named_parameters() if n.startswith("trunk.")
    )
    print(f"FINAL heldout={acc(m, ev):.4f} train_acc={train_acc:.4f} trunk_max_delta={delta:.4f}", flush=True)
    if args.save:
        out = os.path.join(ROOT, "checkpoints_arth_sem_probe", "arth_final.pt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"student": m.state_dict()}, out)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
