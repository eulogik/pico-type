"""RLCD-lite (plan §2/§3): Gaussian-noise group-baseline RL on REWARD-MEASURED
heads only (semantic + riskpp). Trunk is never touched — legacy parity is
structurally preserved (Phase-B trunk RL was rejected; heads-only is the
approved policy 2026-09-23).

Algorithm (per plan: "Gaussian noise + Brier reward + group baseline"):
  for each step:
    sample one minibatch (same for all group members — paired comparison)
    for i in 1..G:  eps_i ~ N(0,1);  theta_i = theta + sigma*eps_i
                    R_i = -(brier_choice + brier_risk) + 0.5*accuracy
    A_i = normalize(R_i - mean(R))                    # group baseline
    theta += eta / (sigma * G) * sum_i A_i * eps_i

Reward buffer = HELD-OUT seeds (70100+, unseen by supervised training seed 7).

Usage: python scripts/arth_rlcd.py [--steps 400 --group 4 --sigma 0.01 --eta 3e-4]
Acceptance (checked by caller): battery parity 60/60, demos 7/7, ToxicChat gate.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arth import ArthModel
from model.pico_type.arth_data import (
    RISKPP_GENERATORS,
    gen_benign_hard,
    make_choice_decisions,
)

RLCD_PREFIXES = ("semantic.", "riskpp.")


def encode(batch: list[str], max_bytes: int = 1024):
    ids = [[b for b in s.encode("utf-8", errors="replace")[:max_bytes]] for s in batch]
    ln = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), ln, dtype=torch.long)
    mask = torch.zeros(len(ids), ln, dtype=torch.bool)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    return pad, mask


def choice_logits(m: ArthModel, items: list[dict]) -> torch.Tensor:
    """(B,K-padded) choice logits, training-consistent (input-only trunk pass)."""
    out = []
    maxk = max(len(it["options"]) for it in items)
    ids, mask = encode([it["input"] for it in items])
    pooled = m.pooled(ids, mask)
    bems = []
    for it in items:
        rows = []
        for o in it["options"]:
            ob = o.encode("utf-8", errors="replace")[:256] or b"\x00"
            rows.append(m.trunk.embed(torch.tensor([[b for b in ob]])).mean(dim=1))
        while len(rows) < maxk:
            rows.append(torch.zeros_like(rows[0]))
        bems.append(torch.cat(rows, dim=0))
    emb = torch.stack(bems)
    logits = m.semantic(pooled, emb)
    kmask = torch.tensor(
        [[1] * len(it["options"]) + [0] * (maxk - len(it["options"])) for it in items],
        dtype=torch.bool,
    )
    logits = logits.masked_fill(~kmask, float("-inf"))
    out.append(logits)
    return out[0]


def reward(
    m: ArthModel,
    sem_items: list[dict],
    risk_items: list[dict],
    sem_idx: list[int],
    risk_idx: list[int],
) -> float:
    """R = -(brier_choice + brier_risk) + 0.5*accuracy (bounded, plan RLCR)."""
    sb = [sem_items[i] for i in sem_idx]
    rb = [risk_items[i] for i in risk_idx]
    with torch.no_grad():
        lg = choice_logits(m, sb)
        tgt = torch.tensor([it["correct"] for it in sb], dtype=torch.long)
        probs = F.softmax(lg, dim=-1)
        onehot = F.one_hot(tgt, probs.shape[-1]).float()
        brier_sem = float(((probs - onehot) ** 2).sum(-1).mean())
        acc = float((probs.argmax(-1) == tgt).float().mean())
        ids, mask = encode([it["input"] for it in rb])
        pooled = m.pooled(ids, mask)
        rp = torch.sigmoid(m.riskpp(pooled))
        rt = torch.tensor([it["risk14"] for it in rb], dtype=torch.float)
        brier_risk = float(((rp - rt) ** 2).mean())
    return -(brier_sem + brier_risk) + 0.5 * acc


def noise_params(m: ArthModel) -> list[tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in m.named_parameters() if n.startswith(RLCD_PREFIXES)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_arth_ft2/arth_final.pt")
    ap.add_argument("--out", default="checkpoints_arth_ft3")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--eta", type=float, default=2e-5)  # ~0.15 sigma/step; 3e-4 collapsed reward
    ap.add_argument("--bs", type=int, default=32)  # per-reward minibatch half
    ap.add_argument("--seed", type=int, default=70100)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    m = ArthModel(os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["student"])
    m.eval()
    trunk_before = {k: v.clone() for k, v in m.trunk.state_dict().items()}

    # held-out reward buffer (seed 70100 — never trained on)
    sem_items = make_choice_decisions(600, args.seed)
    risk_items = [x for g, s in zip(RISKPP_GENERATORS.values(), range(8)) for x in g(40, args.seed + s)]
    risk_items += gen_benign_hard(160, args.seed)
    print(f"buffer: sem={len(sem_items)} risk={len(risk_items)}")

    params = noise_params(m)
    base = [p.detach().clone() for _, p in params]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    def set_params(vecs: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for i, (_, p) in enumerate(params):
                p.copy_(vecs[i])

    t0 = time.time()
    hist = []
    for step in range(args.steps):
        sem_idx = [rng.randrange(len(sem_items)) for _ in range(args.bs)]
        risk_idx = [rng.randrange(len(risk_items)) for _ in range(args.bs)]
        rewards = []
        epsilons = []
        for _ in range(args.group):
            eps = [torch.randn_like(b) for b in base]
            perturbed = [b + args.sigma * e for b, e in zip(base, eps)]
            set_params(perturbed)
            rewards.append(reward(m, sem_items, risk_items, sem_idx, risk_idx))
            epsilons.append(eps)
        r = torch.tensor(rewards)
        adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
        # natural-gradient-ish ES update
        with torch.no_grad():
            for i, (_, p) in enumerate(params):
                acc = torch.zeros_like(base[i])
                for g_i in range(args.group):
                    acc += float(adv[g_i]) * epsilons[g_i][i]
                base[i] = base[i] + (args.eta / (args.sigma * args.group)) * acc
                p.copy_(base[i])
        hist.append(float(r.mean()))
        if step % args.log_every == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"step {step}: R_mean={float(r.mean()):+.4f} R=[{min(rewards):+.3f},{max(rewards):+.3f}] t={el:.0f}s")

    # trunk lock proof
    for k, v in m.trunk.state_dict().items():
        assert torch.equal(v, trunk_before[k]), f"trunk mutated: {k}"
    print("trunk lock: PASS (byte-identical)")

    # persist in arth_train-compatible format
    os.makedirs(args.out, exist_ok=True)
    final_step = int(sd.get("step", 0)) + args.steps
    from model.pico_type.arth_train import make_optim

    class _A:
        lr = 3e-4
        lr_trunk = 3e-5

    opt = make_optim(m, _A(), unfrozen=False)
    ck = {"step": final_step, "student": m.state_dict(), "optim": opt.state_dict()}
    torch.save(ck, os.path.join(args.out, f"arth_step_{final_step}.pt"))
    torch.save(ck, os.path.join(args.out, "arth_final.pt"))
    print(f"saved {args.out}/arth_final.pt step={final_step}")
    print(f"reward: first5={np.mean(hist[:5]):+.4f} last5={np.mean(hist[-5:]):+.4f}")


if __name__ == "__main__":
    main()
