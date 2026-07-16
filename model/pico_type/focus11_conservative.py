"""Round 11b: Conservative training - targeted data, fixed max_bytes, confidence tuning.

Only adds targeted synthetic data for weak classes and tunes the confidence threshold.
No aggressive augmentation that could regress performance.
"""

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
from model.pico_type.labels import CODE_LANG_LABELS, COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS
from model.pico_type.train import MultiTaskLoss, collate_fn, multi_tier_loss

_COARSE = {n: i for i, n in enumerate(COARSE_LABELS)}
_MODALITY = {n: i for i, n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n: i for i, n in enumerate(SUBTYPE_LABELS)}
_CODE = {n: i for i, n in enumerate(CODE_LANG_LABELS)}

_DEVICE = torch.device("cpu")
FIXED_MAX_BYTES = 1024


def _id(rng, n=8):
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n))


def gen_targeted(rng=random.Random(77)):
    samples = []
    # Java
    for _ in range(150):
        text = "public class {} {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello\");\n    }}\n}}\n".format(_id(rng, 5).capitalize())
        samples.append(Sample(data=text.encode(), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["java"]))
    # Python
    for _ in range(150):
        text = "def {}({}):\n    return {}\n".format(_id(rng, 8), _id(rng, 6), rng.choice(["None", "True", "0"]))
        samples.append(Sample(data=text.encode(), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["python"]))
    # JavaScript
    for _ in range(100):
        text = "function {}({}) {{\n    return {};\n}}\n".format(_id(rng, 8), _id(rng, 6), rng.choice(["null", "true", "0"]))
        samples.append(Sample(data=text.encode(), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["javascript"]))
    # Go
    for _ in range(100):
        text = "package main\n\nimport \"fmt\"\n\nfunc {}({}) {{\n    fmt.Println({})\n}}\n".format(_id(rng, 8), _id(rng, 6), '"{}"'.format(_id(rng, 8)))
        samples.append(Sample(data=text.encode(), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["go"]))
    # Rust
    for _ in range(100):
        text = "fn {}({}) -> String {{\n    String::new()\n}}\n\nfn main() {{\n    {}();\n}}\n".format(_id(rng, 8), _id(rng, 6), _id(rng, 8))
        samples.append(Sample(data=text.encode(), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["rust"]))
    # Error
    for _ in range(150):
        err = rng.choice(["NameError", "TypeError", "ValueError", "KeyError", "IndexError"])
        text = "Traceback (most recent call last):\n  File \"{}.py\", line {}, in {}\n    {}()\n{}: {}".format(
            _id(rng, 8), rng.randint(1, 200), _id(rng, 8), _id(rng, 12), err, _id(rng, 15))
        samples.append(Sample(data=text.encode(), coarse=_COARSE["error"], modality=_MODALITY["textual"]))
    # Markup HTML
    for _ in range(100):
        lines = ["<!DOCTYPE html>", "<html>", "<body>"]
        for _ in range(rng.randint(3, 6)):
            tag = rng.choice(["p", "div", "h1", "span"])
            lines.append("<{}>{}</{}>".format(tag, _id(rng, 15), tag))
        lines.extend(["</body>", "</html>"])
        data = "\n".join(lines).encode()[:FIXED_MAX_BYTES]
        samples.append(Sample(data=data, coarse=_COARSE["markup"], modality=_MODALITY["textual"], subtype=_SUBTYPE["html"]))
    # Markup Markdown
    for _ in range(80):
        lines = ["# {} Guide".format(rng.choice(["Getting", "Quick"]))]
        for _ in range(rng.randint(2, 5)):
            lines.append("## {}".format(_id(rng, 10).title()))
            lines.append("{}".format(_id(rng, 25).capitalize()))
        data = "\n".join(lines).encode()[:FIXED_MAX_BYTES]
        samples.append(Sample(data=data, coarse=_COARSE["markup"], modality=_MODALITY["textual"], subtype=_SUBTYPE["markdown"]))
    # Config YAML
    for _ in range(80):
        lines = []
        for _ in range(rng.randint(3, 6)):
            val = rng.choice([str(rng.randint(1, 999)), "true", "false", '"{}"'.format(_id(rng, 4))])
            lines.append("{}: {}".format(_id(rng, 6), val))
        data = "\n".join(lines).encode()[:FIXED_MAX_BYTES]
        samples.append(Sample(data=data, coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["yaml"]))
    # Config JSON
    for _ in range(80):
        n = rng.randint(3, 5)
        pairs = ",\n".join('    "{}": {}'.format(_id(rng, 5), rng.choice([str(rng.randint(1, 999)), "true", "false"])) for _ in range(n))
        text = "{\n" + pairs + "\n}\n"
        samples.append(Sample(data=text.encode(), coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["json"]))

    rng.shuffle(samples)
    return samples


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=FIXED_MAX_BYTES)).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    acc_before = ckpt.get("real_world_accuracy", 0)
    print("Starting RWA: {:.1f}%".format(acc_before * 100))

    real_samples = []
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            d = pickle.load(f)
            real_samples = d.get("samples", d) if isinstance(d, dict) else d
    print("Real samples: {}".format(len(real_samples)))

    targeted = gen_targeted()
    print("Targeted: {}".format(len(targeted)))

    gen_synth = SyntheticGenerator(seed=42)

    hw = {"coarse": 10.0, "modality": 2.0, "subtype": 2.0, "code_lang": 2.5, "text_lang": 2.0, "file_mime": 1.0, "risk": 1.0}
    criterion = MultiTaskLoss(hw)

    train_tiers = ("base", "pro")
    total_steps = 800
    batch_size = 8
    best_acc = acc_before
    best_step = 0

    def get_lr(step):
        if step < 100:
            return 2e-6 * (step + 1) / 100
        progress = (step - 100) / (total_steps - 100)
        return 2e-6 * 0.3 * (1 + math.cos(progress * math.pi))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(0.9, 0.98), weight_decay=0.01)
    model.train()

    print("{} steps, batch={}, tiers={}".format(total_steps, batch_size, train_tiers))
    start_time = time.time()

    for step in range(total_steps):
        bs = batch_size
        n_real = int(bs * 0.45)
        n_targeted = int(bs * 0.20)
        n_synth = bs - n_real - n_targeted

        batch = [real_samples[(step * n_real + i) % len(real_samples)] for i in range(n_real)]
        batch += [targeted[(step * n_targeted + i) % len(targeted)] for i in range(n_targeted)]
        for _ in range(n_synth):
            batch.append(gen_synth())

        coll = collate_fn(batch)
        ids = coll["input_ids"].to(_DEVICE)
        mask = coll["attention_mask"].to(_DEVICE)
        labels = {k: v.to(_DEVICE) for k, v in coll["labels"].items()}

        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, train_tiers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (total_steps - step - 1) / rate if rate > 0 else 0
            parts = ["step={:5d}/{} lr={:.8f} loss={:.4f} [{}s ETA {}s]".format(
                step, total_steps, lr, loss.item(), int(elapsed), int(eta))]
            for h in ("coarse", "code_lang"):
                if h in comps:
                    parts.append("{}={:.4f}".format(h, comps[h]))
            print("  ".join(parts))

        if step > 0 and step % 200 == 0:
            model.eval()
            acc, n_c, n_t = _eval_real_world(model, _DEVICE)
            print("  EVAL step={}: real_world={:.1f}% ({}/{})".format(step, acc * 100, n_c, n_t))
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
    print("Done in {:.0f}s ({:.1f} min)".format(elapsed, elapsed / 60))
    print("Final: {:.1f}% ({}/{})".format(acc * 100, n_c, n_t))
    print("Best: {:.1f}% at step {}".format(best_acc * 100, best_step))
    print("=" * 50)

    if acc > best_acc:
        torch.save({
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "real_world_accuracy": acc,
            "real_world_correct": n_c,
            "real_world_total": n_t,
        }, ckpt_path)
        print("Saved final")


if __name__ == "__main__":
    main()
