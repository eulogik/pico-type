"""Continue sweep from step 600 to 1200."""

from __future__ import annotations

import math
import os
import pickle
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.data import SyntheticGenerator, Sample
from model.pico_type.finetune_real import _eval_real_world
from model.pico_type.labels import (
    CODE_LANG_LABELS, COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS, TEXT_LANG_LABELS,
)
from model.pico_type.train import MultiTaskLoss, collate_fn, multi_tier_loss

_COARSE = {n: i for i, n in enumerate(COARSE_LABELS)}
_MODALITY = {n: i for i, n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n: i for i, n in enumerate(SUBTYPE_LABELS)}
_CODE = {n: i for i, n in enumerate(CODE_LANG_LABELS)}
_TEXT = {n: i for i, n in enumerate(TEXT_LANG_LABELS)}
_DEVICE = torch.device("cpu")


def _id(rng, n=8):
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n))


def gen_targeted(rng=random.Random(77)):
    samples = []
    java_templates = [
        "public class {cls} {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello\");\n    }}\n}}\n",
    ]
    for _ in range(200):
        text = rng.choice(java_templates).format(cls=_id(rng, 5).capitalize())
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["java"]))

    for _ in range(200):
        n = rng.randint(3, 6)
        keys = [_id(rng, 5) for _ in range(n)]
        vals = [str(rng.randint(1, 999)), _id(rng, 4), rng.choice(["true", "false"]), str(rng.randint(10, 9999))]
        pairs = ",\n".join('    "{}": "{}"'.format(k, v) for k, v in zip(keys, vals))
        text = "{\n" + pairs + "\n}\n"
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["json"]))

    for _ in range(100):
        text = "name: {}\nversion: {}\ndescription: {}\n".format(_id(rng, 5), rng.randint(1, 999), _id(rng, 6))
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["yaml"]))

    hindi = ["नमस्ते दुनिया! आप कैसे हैं?", "भारत एक महान देश है।", "आज मौसम अच्छा है।"]
    for _ in range(100):
        samples.append(Sample(data=rng.choice(hindi).encode("utf-8"), coarse=_COARSE["text"], modality=_MODALITY["textual"], text_lang=_TEXT["hi"]))

    fr = ["Bonjour le monde! Comment allez-vous?", "Je m'appelle Pierre et j'habite a Paris."]
    for _ in range(100):
        samples.append(Sample(data=rng.choice(fr).encode("utf-8"), coarse=_COARSE["text"], modality=_MODALITY["textual"], text_lang=_TEXT["fr"]))

    rng.shuffle(samples)
    return samples


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")
    snapshot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "sweep")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Load from step 600 snapshot
    snap_path = os.path.join(snapshot_dir, "step_00600.pt")
    snap = torch.load(snap_path, map_location="cpu", weights_only=False)

    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(_DEVICE)
    model.load_state_dict(snap["model_state_dict"])
    print("Resumed from step 600")

    real_samples = []
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            d = pickle.load(f)
            real_samples = d.get("samples", d) if isinstance(d, dict) else d
    targeted = gen_targeted()
    gen_synth = SyntheticGenerator(seed=42)
    hw = {"coarse": 10.0, "modality": 2.0, "subtype": 2.0, "code_lang": 1.5, "text_lang": 1.5, "file_mime": 1.0, "risk": 1.0}
    criterion = MultiTaskLoss(hw)

    total_steps = 1200
    start_step = 600

    def get_lr(step):
        if step < 200:
            return 1e-6 * (step + 1) / 200
        progress = (step - 200) / (total_steps - 200)
        return 1e-6 * 0.3 * (1 + math.cos(progress * math.pi))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    model.train()

    start_time = time.time()
    for step in range(start_step, total_steps):
        bs = 16
        n_real = int(bs * 0.70)
        n_targeted = int(bs * 0.15)
        n_synth = bs - n_real - n_targeted

        batch = [real_samples[(step * n_real + i) % len(real_samples)] for i in range(n_real)]
        batch += [targeted[(step * n_targeted + i) % len(targeted)] for i in range(n_targeted)]
        batch += [gen_synth() for _ in range(n_synth)]

        coll = collate_fn(batch)
        ids = coll["input_ids"].to(_DEVICE)
        mask = coll["attention_mask"].to(_DEVICE)
        labels = {k: v.to(_DEVICE) for k, v in coll["labels"].items()}

        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, ("base",))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            snapshot_path = os.path.join(snapshot_dir, "step_{:05d}.pt".format(step))
            torch.save({"model_state_dict": model.state_dict(), "step": step}, snapshot_path)

        if step % 200 == 0:
            elapsed = time.time() - start_time
            print("step={:5d} lr={:.8f} loss={:.4f} [{}s]".format(step, lr, loss.item(), int(elapsed)))

    # Sweep steps 600-1200
    print("\nSWEEP 600-1200:")
    best_acc = 0.905  # current baseline
    best_step = 0
    best_snapshot = None

    for step in range(100, total_steps + 1, 100):
        snapshot_path = os.path.join(snapshot_dir, "step_{:05d}.pt".format(step))
        if not os.path.exists(snapshot_path):
            continue
        snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        model.load_state_dict(snap["model_state_dict"])
        model.eval()
        acc, cor, tot = _eval_real_world(model, _DEVICE)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_step = step
            best_snapshot = snapshot_path
            marker = " *** NEW BEST ***"
        print("  step={:5d}: {:.1f}% ({}/{}){}".format(step, acc * 100, cor, tot, marker))

    if best_snapshot and best_acc > 0.905:
        snap = torch.load(best_snapshot, map_location="cpu", weights_only=False)
        torch.save({
            "model_state_dict": snap["model_state_dict"],
            "step": best_step,
            "config": model.config.__dict__,
            "real_world_accuracy": best_acc,
        }, ckpt_path)
        print("Saved best.pt: {:.1f}%".format(best_acc * 100))

    # Cleanup
    for f in os.listdir(snapshot_dir):
        os.remove(os.path.join(snapshot_dir, f))
    os.rmdir(snapshot_dir)
    print("Done")


if __name__ == "__main__":
    main()
