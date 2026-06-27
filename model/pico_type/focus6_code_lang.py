"""Round 6: Focused training on code language detection with real data."""

from __future__ import annotations

import math
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.labels import CODE_LANG_LABELS
from model.pico_type.train import multi_tier_loss, MultiTaskLoss, collate_fn

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _load_real_samples(path: str) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["samples"] if isinstance(data, dict) else data


def _augment(sample):
    d = sample.data
    kind = random.random()
    if kind < 0.2:
        d = d[:max(32, len(d) // 2)]
    elif kind < 0.4:
        d = d[len(d) // 4:]
    elif kind < 0.6:
        prefix = b"#!/usr/bin/env python3\n\n"
        d = prefix + d[:2048 - len(prefix)]
    elif kind < 0.8:
        d = b"\n".join(line.rstrip() for line in d.split(b"\n") if line.strip())
    from model.pico_type.data import Sample
    return Sample(data=d[:2048], coarse=sample.coarse, modality=sample.modality,
                  code_lang=sample.code_lang)


def eval_code_lang_accuracy(model, real_samples, device):
    model.eval()
    correct = 0
    total = 0
    lang_correct = {}
    lang_total = {}
    with torch.no_grad():
        for s in real_samples:
            if s.coarse != 2:
                continue
            data = s.data[:2048]
            padding = b"\x00" * (2048 - len(data))
            ids = torch.tensor([list(data) + list(padding[:2048 - len(data)])], dtype=torch.long, device=device)
            mask = torch.tensor([[1] * len(data) + [0] * (2048 - len(data))], dtype=torch.bool, device=device)
            out = model(ids, mask, tier="base")
            pred = out["code_lang"][0].argmax(-1).item()
            total += 1
            lang = CODE_LANG_LABELS[s.code_lang]
            lang_total[lang] = lang_total.get(lang, 0) + 1
            if pred == s.code_lang:
                correct += 1
                lang_correct[lang] = lang_correct.get(lang, 0) + 1
    acc = correct / total if total else 0
    return acc, lang_correct, lang_total


def main():
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")
    real_samples = _load_real_samples(real_path)
    code_samples = [s for s in real_samples if s.coarse == 2]
    print(f"Loaded {len(code_samples)} code samples from {len(real_samples)} total")

    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt.get('real_world_accuracy', 0)*100:.1f}%")

    acc_before, lc, lt = eval_code_lang_accuracy(model, real_samples, _DEVICE)
    print(f"\nCode-lang accuracy BEFORE: {acc_before*100:.1f}%")
    for lang in sorted(lt.keys()):
        c = lc.get(lang, 0)
        t = lt[lang]
        print(f"  {lang:12s}: {c}/{t} ({c/t*100:.0f}%)")

    criterion = MultiTaskLoss({"coarse": 8.0})
    lr = 5e-6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps = 2000

    model.train()
    step = 0
    best_acc = acc_before
    best_state = model.state_dict()

    print(f"\nTraining: {len(code_samples)} code samples, lr={lr}, steps={total_steps}")

    while step < total_steps:
        batch = random.choices(code_samples, k=16)
        batch = [_augment(s) for s in batch]
        coll = collate_fn(batch)
        ids = coll["input_ids"].to(_DEVICE)
        mask = coll["attention_mask"].to(_DEVICE)
        labels = {k: v.to(_DEVICE) for k, v in coll["labels"].items()}

        cur_lr = lr * min(1, (step + 1) / 100) if step < 100 else lr * 0.5 * (1 + math.cos((step - 100) / (total_steps - 100) * math.pi))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, ("base",))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            parts = [f"step={step:5d} lr={cur_lr:.7f} loss={loss.item():.4f}"]
            for h in ("coarse", "modality", "code_lang"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        if step % 200 == 0 and step > 0:
            model.eval()
            acc, lc, lt = eval_code_lang_accuracy(model, real_samples, _DEVICE)
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                print(f"  *** NEW BEST code-lang: {acc*100:.1f}% ***")
            model.train()

        step += 1

    model.load_state_dict(best_state)
    model.eval()
    final_acc, lc, lt = eval_code_lang_accuracy(model, real_samples, _DEVICE)
    print(f"\nFinal code-lang accuracy: {final_acc*100:.1f}%")
    for lang in sorted(lt.keys()):
        c = lc.get(lang, 0)
        t = lt[lang]
        print(f"  {lang:12s}: {c}/{t} ({c/t*100:.0f}%)")

    if final_acc > acc_before:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": step,
            "config": model.config.__dict__,
            "real_world_accuracy": final_acc,
        }, ckpt_path)
        print(f"\nSaved improved checkpoint: {ckpt_path}")
    else:
        print(f"\nNo improvement ({final_acc*100:.1f}% <= {acc_before*100:.1f}%), keeping original.")


if __name__ == "__main__":
    main()
