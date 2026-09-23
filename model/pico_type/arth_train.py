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
    if not batch_texts:
        raise ValueError("encode() called with empty batch")
    ids = [[b for b in t[:max_bytes]] for t in batch_texts]
    ln = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), ln, dtype=torch.long)
    mask = torch.zeros(len(ids), ln, dtype=torch.bool)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    return pad, mask


def latest_ckpt(out_dir: str) -> str | None:
    """Numeric (not lexicographic) latest arth_step_*.pt — 'arth_step_100' > 'arth_step_50'."""
    best, best_n = None, -1
    if not os.path.isdir(out_dir):
        return None
    for f in os.listdir(out_dir):
        if f.startswith("arth_step_") and f.endswith(".pt"):
            try:
                n = int(f[len("arth_step_") : -len(".pt")])
            except ValueError:
                continue
            if n > best_n:
                best, best_n = os.path.join(out_dir, f), n
    return best


def make_optim(student: ArthModel, args, unfrozen: bool) -> torch.optim.Optimizer:
    """Group structure MUST match what was saved (1 group Phase A, 2 after unfreeze)."""
    heads = [p for n, p in student.named_parameters() if not n.startswith("trunk.")]
    groups: list[dict] = [{"params": heads, "lr": args.lr, "weight_decay": 0.01}]
    if unfrozen:
        trunk = [p for n, p in student.named_parameters() if n.startswith("trunk.")]
        groups.append({"params": trunk, "lr": args.lr_trunk, "weight_decay": 0.01})
    return torch.optim.AdamW(groups)


def unfreeze_trunk(student: ArthModel) -> None:
    for p in student.trunk.parameters():
        p.requires_grad_(True)
    student.trunk.train()


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
        benign = load_jsonl("benign_hard", subset)
        # mix (FP-audit): benign x1 — x2 diluted prompt_injection recall (0.58)
        self.risk = self.risk + benign
        # real ToxicChat train-split jailbreak signal (gate fix; test split held out)
        self.risk = self.risk + load_jsonl("toxicchat_jail", subset)
        self.choice = load_jsonl("semantic_choice", subset)
        self.score = load_jsonl("semantic_score", subset)
        self.noul = load_jsonl("semantic_noul", subset)
        self.structural = load_jsonl("structural", subset)
        self.invoice = load_jsonl("invoice", subset)
        self.unans = load_jsonl("unanswerable", subset)
        # pre-registered external tables (AG/SST-2/Enron gates, plan §3)
        self.ext = (
            load_jsonl("ag_news", subset)
            + load_jsonl("sst2", subset)
            + load_jsonl("enron_spam", subset)
        )
        heap = load_heap_code()
        wiki = load_wiki_text()
        if subset:
            heap = heap[:subset]
            wiki = wiki[:subset]
        self.legacy = [b for b, _ in heap] + [b for b, _ in wiki]
        # choice x2: protect in-domain share now that ext tables join the pool
        self.sem_pool = self.choice * 2 + self.score + self.noul + self.invoice + self.unans + self.ext

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

    # Resume FIRST, then build optimizer with the saved group structure:
    # post-unfreeze ckpts have 2 param_groups; loading them into a 1-group
    # optimizer raises ValueError. Saved group count is the source of truth.
    ck = None
    ckpt_path = latest_ckpt(args.out) if args.resume else None
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"])

    start = (ck["step"] + 1) if ck else 0
    unfrozen = bool(ck and len(ck["optim"]["param_groups"]) > 1)
    if unfrozen:
        unfreeze_trunk(student)
        print(f"resume: trunk already unfrozen ({os.path.basename(ckpt_path)})")

    opt = make_optim(student, args, unfrozen)
    if ck:
        opt.load_state_dict(ck["optim"])
        print(f"resumed from step {start}")

    hist: list[dict] = []
    os.makedirs(args.out, exist_ok=True)

    for step in range(start, args.steps):
        if step >= args.freeze_steps and not unfrozen:
            unfreeze_trunk(student)
            opt.add_param_group(
                {"params": [p for n, p in student.named_parameters() if n.startswith("trunk.")],
                 "lr": args.lr_trunk, "weight_decay": 0.01}
            )
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
            opt_bytes = []
            offsets = []
            cursor = 0
            for item in rel:
                offsets.append(cursor)
                for o in item["options"]:
                    opt_bytes.append(o.encode("utf-8", errors="replace")[:512])
                    cursor += 1
            oids, omask = encode(opt_bytes, max_bytes=512)
            oids, omask = oids.to(device), omask.to(device)
            op = student.pooled(oids, omask)
            feats = torch.tensor(
                np.stack([structural_features(b) for b in opt_bytes]),
                dtype=torch.float,
                device=device,
            )
            scores = student.relational(op, feats)
            l_rel = torch.zeros((), device=device)
            for i, item in enumerate(rel):
                k = len(item["options"])
                grp = scores[offsets[i] : offsets[i] + k].unsqueeze(0)
                l_rel = l_rel + F.cross_entropy(grp, torch.tensor([item["correct"]], device=device))
            losses["rel"] = l_rel / len(rel)

        total = (
            args.w_legacy * losses.get("legacy", torch.zeros(()))
            + args.w_sem * losses.get("sem", torch.zeros(()))
            + args.w_rel * losses.get("rel", torch.zeros(()))
            + args.w_risk * losses.get("risk", torch.zeros(()))
            + args.lambda_cal * losses.get("cal", torch.zeros(()))
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"non-finite loss at step {step}: { {k: float(v.detach()) for k, v in losses.items()} }"
            )
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
    if not h:
        print("no steps run (resume start >= --steps?)")
        return
    print(f"done: {len(h)} steps, total {h[0]['total']:.4f} -> {h[-1]['total']:.4f}")


if __name__ == "__main__":
    main()
