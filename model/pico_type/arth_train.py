"""ARTH supervised training: frozen-target KD + semantic/relational/risk/cal losses.

Phase A (steps < freeze_steps): trunk frozen, only new heads train.
Phase B (steps >= freeze_steps): trunk unfrozen (small LR), KD guards legacy.

  L = 1.0*L_legacyKD + 1.0*L_sem + 0.5*L_rel + 0.5*L_risk + 0.005*L_calCE   (plan §2)

Streams (mixed batch): legacy KD inputs (Heap/Wiki/synth bytes, teacher=soft
targets), semantic decisions (choice CE / score BCE / noul BCE-zeros),
structural choice (relational per-option CE), risk14 (BCE).

Usage:
  python -m model.pico_type.arth_train --steps 60 --subset 200   # smoke test
  python -m model.pico_type.arth_train --steps 3000              # real run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from model.pico_type.arth import ArthModel, structural_features
from model.pico_type.arth_data import load_heap_code, load_wiki_text
from model.pico_type.labels import ALL_HEADS

TRUNK_CKPT = os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt")
DATA_DIR = os.path.join(ROOT, "data", "raw", "arth")
MAX_BYTES = 1024


def encode(batch_texts: list[bytes], max_bytes: int = MAX_BYTES):
    ids = [[b for b in t[:max_bytes]] for t in batch_texts]
    ln = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), ln, dtype=torch.long)
    mask = torch.zeros(len(ids), ln, dtype=torch.bool)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    return pad, mask


def load_jsonl(name: str, subset: int = 0) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if subset and len(out) >= subset:
                break
    return out


class Streams:
    def __init__(self, seed: int = 7, subset: int = 0):
        self.rng = random.Random(seed)
        self.risk = load_jsonl("riskpp_synth", subset)
        self.choice = load_jsonl("semantic_choice", subset)
        self.score = load_jsonl("semantic_score", subset)
        self.noul = load_jsonl("semantic_noul", subset)
        self.structural = load_jsonl("structural", subset)
        self.invoice = load_jsonl("invoice", subset)
        self.unans = load_jsonl("unanswerable", subset)
        heap = load_heap_code()
        wiki = load_wiki_text()
        if subset:
            heap = heap[:subset]
            wiki = wiki[:subset]
        self.legacy = [b for b, _ in heap] + [b for b, _ in wiki]
        self.sem_pool = self.choice + self.score + self.noul + self.invoice + self.unans

    def sample(self, n_legacy: int, n_sem: int, n_risk: int, n_rel: int):
        leg = self.rng.sample(self.legacy, min(n_legacy, len(self.legacy)))
        sem = self.rng.sample(self.sem_pool, min(n_sem, len(self.sem_pool)))
        rsk = self.rng.sample(self.risk, min(n_risk, len(self.risk)))
        rel = self.rng.sample(self.structural, min(n_rel, len(self.structural)))
        return leg, sem, rsk, rel


def kd_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], temp: float = 2.0) -> torch.Tensor:
    tot, n = 0.0, 0
    for h in ALL_HEADS:
        s, t = student[h], teacher[h]
        if h == "risk":
            tot = tot + F.binary_cross_entropy_with_logits(s, torch.sigmoid(t))
        else:
            tot = tot + F.kl_div(
                F.log_softmax(s / temp, dim=-1), F.softmax(t / temp, dim=-1), reduction="batchmean"
            ) * (temp * temp)
        n += 1
    return tot / n


def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    student = ArthModel(TRUNK_CKPT).to(device)
    teacher = ArthModel(TRUNK_CKPT).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    streams = Streams(seed=args.seed, subset=args.subset)

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    unfrozen = False
    hist: list[dict] = []
    os.makedirs(args.out, exist_ok=True)

    start = 0
    ckpts = sorted(f for f in os.listdir(args.out) if f.startswith("arth_step_") and f.endswith(".pt"))
    if args.resume and ckpts:
        ck = torch.load(os.path.join(args.out, ckpts[-1]), map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["optim"])
        start = ck["step"] + 1
        if start >= args.freeze_steps and not unfrozen:
            for p in student.trunk.parameters():
                p.requires_grad_(True)
            opt.add_param_group({"params": list(student.trunk.parameters()), "lr": args.lr_trunk})
            unfrozen = True
        print(f"resumed from step {start}")

    for step in range(start, args.steps):
        if step >= args.freeze_steps and not unfrozen:
            for p in student.trunk.parameters():
                p.requires_grad_(True)
            opt.add_param_group({"params": list(student.trunk.parameters()), "lr": args.lr_trunk})
            unfrozen = True
            print(f"step {step}: trunk unfrozen (lr={args.lr_trunk})")

        leg, sem, rsk, rel = streams.sample(args.n_legacy, args.n_sem, args.n_risk, args.n_rel)
        opt.zero_grad()
        losses: dict[str, torch.Tensor] = {}

        # --- legacy KD (all inputs through one trunk forward where possible) ---
        if leg:
            ids, mask = encode(leg)
            ids, mask = ids.to(device), mask.to(device)
            with torch.no_grad():
                t_logits = teacher.trunk(ids, mask, tier="base")
            s_logits = student.trunk(ids, mask, tier="base")
            losses["legacy"] = kd_loss(s_logits, t_logits, args.kd_temp)

        # --- semantic + risk share one trunk forward ---
        sem_inputs = [s["input"].encode("utf-8", errors="replace")[:MAX_BYTES] for s in sem]
        rsk_inputs = [r["input"].encode("utf-8", errors="replace")[:MAX_BYTES] for r in rsk]
        joint = sem_inputs + rsk_inputs
        if joint:
            jids, jmask = encode(joint)
            jids, jmask = jids.to(device), jmask.to(device)
            pooled = student.pooled(jids, jmask)
            p_sem, p_rsk = pooled[: len(sem_inputs)], pooled[len(sem_inputs) :]

            if sem:
                # option embeds via frozen byte embed
                opt_lists = [s["options"] for s in sem]
                maxk = max(len(o) for o in opt_lists)
                bems, kmask = [], []
                for opts in opt_lists:
                    rows = []
                    for o in opts:
                        ob = o.encode("utf-8", errors="replace")[:256] or b"\x00"
                        rows.append(student.trunk.embed(torch.tensor([[b for b in ob]])).mean(dim=1))
                    while len(rows) < maxk:
                        rows.append(torch.zeros_like(rows[0]))
                    bems.append(torch.cat(rows, dim=0))
                    kmask.append([1] * len(opts) + [0] * (maxk - len(opts)))
                bems_t = torch.stack(bems).to(device)
                kmask_t = torch.tensor(kmask, dtype=torch.bool, device=device)
                logits = student.semantic(p_sem, bems_t)
                logits = logits.masked_fill(~kmask_t, -1e9)
                l_sem = torch.zeros((), device=device)
                n_sem_loss = 0
                cal_terms = []
                for i, s in enumerate(sem):
                    if s["mode"] == "choice":
                        l_sem = l_sem + F.cross_entropy(logits[i : i + 1], torch.tensor([s["correct"]], device=device))
                        n_sem_loss += 1
                        with torch.no_grad():
                            pred = int(logits[i].argmax())
                        if pred != s["correct"]:
                            cal_terms.append(-(F.log_softmax(logits[i : i + 1], dim=-1) + 1e-12).mean())
                    elif s["mode"] == "score":
                        tgt = torch.tensor(s["scores"], dtype=torch.float, device=device)
                        l_sem = l_sem + F.binary_cross_entropy_with_logits(logits[i, : len(tgt)], tgt)
                        n_sem_loss += 1
                    else:  # noul -> abstain: all options low
                        l_sem = l_sem + F.binary_cross_entropy_with_logits(
                            logits[i], torch.zeros_like(logits[i])
                        )
                        n_sem_loss += 1
                losses["sem"] = l_sem / max(1, n_sem_loss)
                losses["cal"] = (sum(cal_terms) / len(cal_terms) if cal_terms else torch.zeros((), device=device))

            if rsk:
                tgt = torch.tensor([r["risk14"] for r in rsk], dtype=torch.float, device=device)
                losses["risk"] = F.binary_cross_entropy_with_logits(student.riskpp(p_rsk), tgt)

        # --- relational: score each option independently ---
        if rel:
            opt_bytes, opt_owner = [], []
            for j, item in enumerate(rel):
                for o in item["options"]:
                    opt_bytes.append(o.encode("utf-8", errors="replace")[:512])
                    opt_owner.append((j, item["correct"]))
            oids, omask = encode(opt_bytes, max_bytes=512)
            oids, omask = oids.to(device), omask.to(device)
            op = student.pooled(oids, omask)
            feats = torch.tensor(
                np.stack([structural_features(b) for b in opt_bytes]),
                dtype=torch.float,
                device=device,
            )
            scores = student.relational(op, feats)
            # group per item (4 options each)
            l_rel = torch.zeros((), device=device)
            for j, item in enumerate(rel):
                grp = scores[j * 4 : j * 4 + 4].unsqueeze(0)
                l_rel = l_rel + F.cross_entropy(grp, torch.tensor([item["correct"]], device=device))
            losses["rel"] = l_rel / len(rel)

        total = (
            args.w_legacy * losses.get("legacy", torch.zeros(()))
            + args.w_sem * losses.get("sem", torch.zeros(()))
            + args.w_rel * losses.get("rel", torch.zeros(()))
            + args.w_risk * losses.get("risk", torch.zeros(()))
            + args.lambda_cal * losses.get("cal", torch.zeros(()))
        )
        assert torch.isfinite(total), f"non-finite loss at step {step}: { {k: float(v) for k, v in losses.items()} }"
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            rec = {"step": step, "total": float(total.detach()), **{k: float(v.detach()) for k, v in losses.items()}}
        hist.append(rec)
        if (step + 1) % args.log_every == 0:
            print(f"step {step + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "step"))
        if (step + 1) % args.save_every == 0:
            torch.save({"step": step, "student": student.state_dict(), "optim": opt.state_dict()},
                       os.path.join(args.out, f"arth_step_{step + 1}.pt"))

    torch.save({"step": args.steps - 1, "student": student.state_dict(), "optim": opt.state_dict()},
               os.path.join(args.out, "arth_final.pt"))
    return {"history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--freeze-steps", type=int, default=2000)
    ap.add_argument("--n-legacy", type=int, default=8)
    ap.add_argument("--n-sem", type=int, default=8)
    ap.add_argument("--n-risk", type=int, default=8)
    ap.add_argument("--n-rel", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-trunk", type=float, default=3e-5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--w-legacy", type=float, default=1.0)
    ap.add_argument("--w-sem", type=float, default=1.0)
    ap.add_argument("--w-rel", type=float, default=0.5)
    ap.add_argument("--w-risk", type=float, default=0.5)
    ap.add_argument("--lambda-cal", type=float, default=0.005)
    ap.add_argument("--out", default="checkpoints_arth")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = train(args)
    h = out["history"]
    print(f"done: {len(h)} steps, total {h[0]['total']:.4f} -> {h[-1]['total']:.4f}")


if __name__ == "__main__":
    main()
