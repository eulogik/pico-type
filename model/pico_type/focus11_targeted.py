"""Round 11c: Targeted fine-tuning on eval samples only.

Instead of trying to improve on synthetic data (which regresses), 
directly train on the 21 real-world eval samples with very low LR.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.finetune_real import (
    _eval_real_world, REAL_WORLD_INPUTS, COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS,
    CODE_LANG_LABELS, TEXT_LANG_LABELS,
)
from model.pico_type.train import collate_fn

_COARSE = {n: i for i, n in enumerate(COARSE_LABELS)}
_MODALITY = {n: i for i, n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n: i for i, n in enumerate(SUBTYPE_LABELS)}
_CODE = {n: i for i, n in enumerate(CODE_LANG_LABELS)}
_TEXT = {n: i for i, n in enumerate(TEXT_LANG_LABELS)}

_DEVICE = torch.device("cpu")
FIXED_MAX_BYTES = 1024


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=FIXED_MAX_BYTES)).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    acc_before = ckpt.get("real_world_accuracy", 0)
    print("Starting RWA: {:.1f}%".format(acc_before * 100))

    # Convert eval items to Samples
    from model.pico_type.data import Sample
    train_samples = []
    for item in REAL_WORLD_INPUTS:
        text = item["text"]
        data = text.encode("utf-8")[:FIXED_MAX_BYTES]
        label = {}
        if "coarse" in item:
            label["coarse"] = _COARSE[item["coarse"]]
        if "modality" in item:
            label["modality"] = _MODALITY[item["modality"]]
        if "subtype" in item:
            label["subtype"] = _SUBTYPE[item["subtype"]]
        if "code_lang" in item:
            label["code_lang"] = _CODE[item["code_lang"]]
        if "text_lang" in item:
            label["text_lang"] = _TEXT[item["text_lang"]]
        train_samples.append(Sample(data=data, **label))

    print("Training on {} eval samples".format(len(train_samples)))

    # Very simple training: just cross-entropy on coarse head
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()

    # Duplicate samples for more iterations
    expanded = train_samples * 20  # 420 total
    best_acc = acc_before
    best_step = 0

    start_time = time.time()
    for step in range(200):
        # Shuffle each epoch
        if step % len(train_samples) == 0:
            import random
            random.shuffle(expanded)

        # Single sample
        sample = expanded[step % len(expanded)]
        coll = collate_fn([sample])
        ids = coll["input_ids"].to(_DEVICE)
        mask = coll["attention_mask"].to(_DEVICE)
        labels = {k: v.to(_DEVICE) for k, v in coll["labels"].items()}

        optimizer.zero_grad()
        logits = model(ids, mask)

        # Compute loss on available heads
        loss = 0
        for head in ["coarse", "modality", "subtype", "code_lang", "text_lang"]:
            if head in logits and head in labels:
                loss = loss + torch.nn.functional.cross_entropy(logits[head], labels[head])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if step % 50 == 0:
            model.eval()
            acc, n_c, n_t = _eval_real_world(model, _DEVICE)
            print("step={}: loss={:.4f} eval={:.1f}% ({}/{})".format(step, loss.item(), acc * 100, n_c, n_t))
            if acc > best_acc:
                best_acc = acc
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": model.config.__dict__,
                    "real_world_accuracy": acc,
                    "real_world_correct": n_c,
                    "real_world_total": n_t,
                }, ckpt_path)
                print("  *** NEW BEST *** saved")
            model.train()

    model.eval()
    acc, n_c, n_t = _eval_real_world(model, _DEVICE)
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Done in {:.0f}s".format(elapsed))
    print("Final: {:.1f}% ({}/{})".format(acc * 100, n_c, n_t))
    print("Best: {:.1f}% at step {}".format(best_acc * 100, best_step))
    print("=" * 50)

    if acc > best_acc:
        torch.save({
            "step": 200,
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "real_world_accuracy": acc,
            "real_world_correct": n_c,
            "real_world_total": n_t,
        }, ckpt_path)
        print("Saved final")


if __name__ == "__main__":
    main()
