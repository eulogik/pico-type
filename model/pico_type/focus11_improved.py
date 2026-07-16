"""Round 11: Improved training with augmentation, fixed max_bytes.

Improvements over focus10:
1. max_bytes=1024 (consistent with model config)
2. Data augmentation (byte flip, truncation, case swap, noise insertion)
3. Base + pro tiers trained
4. More targeted synthetic data for weak classes
5. Better LR schedule
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
from model.pico_type.data import SyntheticGenerator, AugmentedGenerator, Sample
from model.pico_type.finetune_real import _eval_real_world
from model.pico_type.labels import (
    CODE_LANG_LABELS, COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS,
    TEXT_LANG_LABELS,
)
from model.pico_type.train import MultiTaskLoss, collate_fn, multi_tier_loss

_COARSE = {n: i for i, n in enumerate(COARSE_LABELS)}
_MODALITY = {n: i for i, n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n: i for i, n in enumerate(SUBTYPE_LABELS)}
_CODE = {n: i for i, n in enumerate(CODE_LANG_LABELS)}
_TEXT = {n: i for i, n in enumerate(TEXT_LANG_LABELS)}

_DEVICE = torch.device("cpu")
FIXED_MAX_BYTES = 1024


def _id(rng, n=8):
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n))


def gen_targeted(rng=random.Random(77)):
    samples = []

    # Java code
    java_templates = [
        "public class {cls} {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello\");\n    }}\n}}\n",
        "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {cls} {{\n    public static void main(String[] args) {{\n        List<String> items = new ArrayList<>();\n        items.add(\"test\");\n    }}\n}}\n",
    ]
    for _ in range(200):
        text = rng.choice(java_templates).format(cls=_id(rng, 5).capitalize())
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["java"]))

    # Python
    py_templates = [
        "def {func}({args}):\n    return {val}\n",
        "import {mod}\n\nclass {cls}:\n    def __init__(self, {args}):\n        self.{attr} = {val}\n",
        "for {var} in range(10):\n    {func}({var})\n",
    ]
    for _ in range(200):
        text = rng.choice(py_templates).format(
            func=_id(rng, 8), args=_id(rng, 6), val=rng.choice(["None", "True", "0"]),
            mod=_id(rng, 5), cls=_id(rng, 5).title(), attr=_id(rng, 6), var=_id(rng, 4),
        )
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["python"]))

    # JavaScript
    js_templates = [
        "function {func}({args}) {{\n    return {val};\n}}\n",
        "const {var} = ({args}) => {{\n    console.log({var});\n}};\n",
        "class {cls} {{\n    constructor({args}) {{\n        this.{attr} = {val};\n    }}\n}}\n",
    ]
    for _ in range(150):
        text = rng.choice(js_templates).format(
            func=_id(rng, 8), args=_id(rng, 6), val=rng.choice(["null", "true", "0"]),
            var=_id(rng, 4), cls=_id(rng, 5).title(), attr=_id(rng, 6),
        )
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["javascript"]))

    # Go
    go_templates = [
        "package main\n\nimport \"fmt\"\n\nfunc {func}({args}) {{\n    fmt.Println({val})\n}}\n",
        "package main\n\ntype {cls} struct {{\n    {var} string\n}}\n",
    ]
    for _ in range(150):
        text = rng.choice(go_templates).format(
            func=_id(rng, 8), args=_id(rng, 6), val=f"\"{_id(rng, 8)}\"",
            cls=_id(rng, 5).title(), var=_id(rng, 4),
        )
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["go"]))

    # Rust
    rust_templates = [
        "fn {func}({args}) -> String {{\n    String::new()\n}}\n\nfn main() {{\n    {func}();\n}}\n",
        "struct {cls} {{\n    {var}: String,\n}}\n\nimpl {cls} {{\n    fn new() -> Self {{ Self {{ {var}: String::new() }} }}\n}}\n",
    ]
    for _ in range(150):
        text = rng.choice(rust_templates).format(
            func=_id(rng, 8), args=_id(rng, 6),
            cls=_id(rng, 5).title(), var=_id(rng, 4),
        )
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["rust"]))

    # Error tracebacks
    error_templates = [
        "Traceback (most recent call last):\n  File \"{file}.py\", line {line}, in {func}\n    {code}\n{err}: {msg}",
        "Traceback (most recent call last):\n  File \"{file}.py\", line {line}\n    {code}\n{err}: {msg}\n\nThe above exception was the direct cause of the following exception:\n\n{err2}: {msg}",
        "Error: {msg}\n    at {func} ({file}.js:{line}:{col})",
        "panic: {msg}\n\ngoroutine 1 [running]:\nmain.{func}()\n\t{file}.go:{line}",
    ]
    error_types = ["NameError", "TypeError", "ValueError", "KeyError", "IndexError", "AttributeError",
                   "RuntimeError", "FileNotFoundError", "ImportError", "SyntaxError"]
    for _ in range(200):
        tmpl = rng.choice(error_templates)
        text = tmpl.format(
            file=_id(rng, 8), line=rng.randint(1, 200), func=_id(rng, 8),
            code="    {}()".format(_id(rng, 12)), err=rng.choice(error_types),
            msg=_id(rng, 15), col=rng.randint(1, 50), err2=rng.choice(error_types),
        )
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["error"], modality=_MODALITY["textual"]))

    # Markup: HTML
    for _ in range(150):
        lines = ["<!DOCTYPE html>", "<html>", "<head><title>Page</title></head>", "<body>"]
        for _ in range(rng.randint(3, 8)):
            tag = rng.choice(["p", "div", "span", "h1", "h2", "li", "a"])
            lines.append("<{}>{}</{}>".format(tag, _id(rng, 20), tag))
        lines.extend(["</body>", "</html>"])
        data = "\n".join(lines).encode()[:FIXED_MAX_BYTES]
        samples.append(Sample(data=data, coarse=_COARSE["markup"], modality=_MODALITY["textual"], subtype=_SUBTYPE["html"]))

    # Markup: Markdown
    for _ in range(100):
        lines = ["# {} Guide".format(rng.choice(["Getting", "Quick", "Basic"]))]
        for _ in range(rng.randint(3, 6)):
            lines.append("## {}".format(_id(rng, 12).title()))
            lines.append("")
            lines.append("{}.".format(_id(rng, 30).capitalize()))
            lines.append("")
        data = "\n".join(lines).encode()[:FIXED_MAX_BYTES]
        samples.append(Sample(data=data, coarse=_COARSE["markup"], modality=_MODALITY["textual"], subtype=_SUBTYPE["markdown"]))

    # Configs: YAML
    for _ in range(100):
        lines = []
        for _ in range(rng.randint(4, 8)):
            val = rng.choice([str(rng.randint(1, 999)), _id(rng, 4), "true", "false", '"{}"'.format(_id(rng, 8))])
            lines.append("{}: {}".format(_id(rng, 8), val))
        data = "\n".join(lines).encode()[:FIXED_MAX_BYTES]
        samples.append(Sample(data=data, coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["yaml"]))

    # Configs: JSON
    for _ in range(100):
        n = rng.randint(3, 6)
        keys = [_id(rng, 5) for _ in range(n)]
        vals = [str(rng.randint(1, 999)), '"{}"'.format(_id(rng, 4)), rng.choice(["true", "false"])]
        pairs = ",\n".join('    "{}": {}'.format(k, v) for k, v in zip(keys, vals))
        text = "{\n" + pairs + "\n}\n"
        samples.append(Sample(data=text.encode("utf-8"), coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["json"]))

    rng.shuffle(samples)
    return samples


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=FIXED_MAX_BYTES)).to(_DEVICE)

    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
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

    gen_aug = AugmentedGenerator(seed=42, aug_prob=0.5)
    gen_synth = SyntheticGenerator(seed=42)

    hw = {"coarse": 10.0, "modality": 2.0, "subtype": 2.0, "code_lang": 3.0, "text_lang": 2.0, "file_mime": 1.0, "risk": 1.0}
    criterion = MultiTaskLoss(hw)

    train_tiers = ("base", "pro")
    total_steps = 1500
    batch_size = 8
    best_acc = acc_before
    best_step = 0

    def get_lr(step):
        if step < 200:
            return 3e-6 * (step + 1) / 200
        progress = (step - 200) / (total_steps - 200)
        return 3e-6 * 0.2 * (1 + math.cos(progress * math.pi))

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "linears" not in n], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if "linears" in n], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(param_groups, lr=3e-6, betas=(0.9, 0.98))
    model.train()

    print("{} steps, batch={}, tiers={}, CPU".format(total_steps, batch_size, train_tiers))
    start_time = time.time()

    for step in range(total_steps):
        bs = batch_size
        n_real = int(bs * 0.40)
        n_targeted = int(bs * 0.25)
        n_synth = bs - n_real - n_targeted

        batch = [real_samples[(step * n_real + i) % len(real_samples)] for i in range(n_real)]
        batch += [targeted[(step * n_targeted + i) % len(targeted)] for i in range(n_targeted)]
        for _ in range(n_synth):
            if step % 2 == 0:
                batch.append(gen_aug())
            else:
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
            for h in ("coarse", "modality", "code_lang", "text_lang"):
                if h in comps:
                    parts.append("{}={:.4f}".format(h, comps[h]))
            print("  ".join(parts))

        if step > 0 and step % 500 == 0:
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
        print("Saved final to {}".format(ckpt_path))


if __name__ == "__main__":
    main()
