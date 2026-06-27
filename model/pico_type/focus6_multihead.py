"""Round 6: Multi-head training with code_lang emphasis."""

from __future__ import annotations

import math
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.data import SyntheticGenerator
from model.pico_type.finetune_real import _eval_real_world
from model.pico_type.labels import (
    CODE_LANG_LABELS,
    COARSE_LABELS,
    MODALITY_LABELS,
    SUBTYPE_LABELS,
    TEXT_LANG_LABELS,
)
from model.pico_type.train import MultiTaskLoss, collate_fn, multi_tier_loss

_COARSE = {n: i for i, n in enumerate(COARSE_LABELS)}
_MODALITY = {n: i for i, n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n: i for i, n in enumerate(SUBTYPE_LABELS)}
_CODE = {n: i for i, n in enumerate(CODE_LANG_LABELS)}
_TEXT = {n: i for i, n in enumerate(TEXT_LANG_LABELS)}

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _augment(sample):
    """Light augmentation — no-op mostly."""
    return sample


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    acc_before = ckpt.get("real_world_accuracy", 0)
    print(f"Loaded checkpoint: RWA={acc_before*100:.1f}%")

    acc_before_eval, _, _ = _eval_real_world(model, _DEVICE)
    print(f"Verified RWA: {acc_before_eval*100:.1f}%")

    real_samples = []
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            d = pickle.load(f)
            real_samples = d.get("samples", d) if isinstance(d, dict) else d
        print(f"Loaded {len(real_samples)} real samples")

    gen_synth = SyntheticGenerator(seed=42)

    hw = {
        "coarse": 8.0,
        "modality": 2.0,
        "subtype": 1.0,
        "code_lang": 3.0,
        "text_lang": 2.0,
        "file_mime": 1.0,
        "risk": 1.0,
    }
    criterion = MultiTaskLoss(hw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=0.01)

    total_steps = 2000
    best_acc = acc_before_eval
    best_state = model.state_dict()
    model.train()

    print(f"\nTraining: {total_steps} steps, lr=3e-6, code_lang_weight=3.0")

    for step in range(total_steps):
        bs = 16
        n_real = bs // 2
        n_synth = bs - n_real

        batch = [real_samples[(step * bs + i) % len(real_samples)] for i in range(n_real)]
        batch.extend([gen_synth() for _ in range(n_synth)])

        coll = collate_fn(batch)
        ids = coll["input_ids"].to(_DEVICE)
        mask = coll["attention_mask"].to(_DEVICE)
        labels = {k: v.to(_DEVICE) for k, v in coll["labels"].items()}

        cur_lr = 3e-6 * min(1, (step + 1) / 100) if step < 100 else 3e-6 * 0.5 * (1 + math.cos((step - 100) / (total_steps - 100) * math.pi))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, ("base",))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            parts = [f"step={step:5d} lr={cur_lr:.7f} loss={loss.item():.4f}"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        if step % 500 == 0 and step > 0:
            acc, cor, tot = _eval_real_world(model, _DEVICE)
            print(f"  EVAL step={step}: RWA={acc*100:.1f}% ({cor}/{tot})")
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                print("  NEW BEST!")
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    final_acc, cor, tot = _eval_real_world(model, _DEVICE)
    print(f"\nFinal RWA: {final_acc*100:.1f}% ({cor}/{tot})")

    if final_acc >= acc_before_eval:
        torch.save(
            {
                "model_state_dict": best_state,
                "step": total_steps,
                "config": model.config.__dict__,
                "real_world_accuracy": final_acc,
            },
            ckpt_path,
        )
        print(f"Saved improved checkpoint: {ckpt_path}")
    else:
        print(f"No improvement ({final_acc*100:.1f}% < {acc_before_eval*100:.1f}%), keeping original.")


if __name__ == "__main__":
    main()
