"""Careful fine-tune: low LR, original head weights, real data augmentation."""

from __future__ import annotations

import argparse
import os
import pickle

import torch

from .arch import PicoType, PicoTypeConfig
from .data import SyntheticGenerator
from .labels import (
    COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS,
    CODE_LANG_LABELS, TEXT_LANG_LABELS, FILE_MIME_LABELS,
)
from .train import collate_fn, multi_tier_loss, MultiTaskLoss

REAL_WORLD_INPUTS: list[dict] = [
    {"text": "def hello():\n    print('Hello, world!')", "coarse": "code", "code_lang": "python"},
    {"text": "import os\nimport sys\n\ndef main():\n    print(os.getcwd())\n\nif __name__ == '__main__':\n    main()", "coarse": "code", "code_lang": "python"},
    {"text": "function greet(name) {\n    return `Hello, ${name}!`;\n}\n\nconsole.log(greet('World'));", "coarse": "code", "code_lang": "javascript"},
    {"text": "#include <stdio.h>\n\nint main() {\n    printf(\"hello, world\\n\");\n    return 0;\n}", "coarse": "code", "code_lang": "c"},
    {"text": "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}", "coarse": "code", "code_lang": "java"},
    {"text": "SELECT id, name, email FROM users WHERE active = 1 ORDER BY name;", "coarse": "code", "code_lang": "sql"},
    {"text": "#!/bin/bash\n\nfor f in *.txt; do\n    echo \"Processing $f\"\n    wc -l \"$f\"\ndone", "coarse": "code", "code_lang": "bash"},
    {"text": "fn main() {\n    println!(\"Hello, world!\");\n}", "coarse": "code", "code_lang": "rust"},
    {"text": "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n    res.json({ message: 'Hello' });\n});\n\napp.listen(3000);", "coarse": "code", "code_lang": "javascript"},
    {"text": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"hello world\")\n}", "coarse": "code", "code_lang": "go"},
    {"text": "<!DOCTYPE html>\n<html>\n<body>\n<h1>Hello</h1>\n</body>\n</html>", "coarse": "markup", "code_lang": "html"},
    {"text": "The quick brown fox jumps over the lazy dog. This is a normal English sentence.", "coarse": "text", "text_lang": "en"},
    {"text": "Bonjour le monde! Comment allez-vous aujourd'hui? J'espere que tout va bien.", "coarse": "text", "text_lang": "fr"},
    {"text": "नमस्ते दुनिया! आप कैसे हैं? मुझे उम्मीद है कि सब कुछ ठीक है।", "coarse": "text", "text_lang": "hi"},
    {"text": "{\n  \"name\": \"test\",\n  \"version\": \"1.0.0\"\n}", "coarse": "config", "subtype": "json"},
    {"text": "name: test\nversion: 1.0.0\ndependencies:\n  - torch\n", "coarse": "config", "subtype": "yaml"},
    {"text": "DATABASE_URL=postgres://localhost:5432/mydb\nDEBUG=true\nPORT=8080", "coarse": "config", "subtype": "env"},
    {"text": "Traceback (most recent call last):\n  File \"test.py\", line 5, in <module>\n    main()\nZeroDivisionError: division by zero", "coarse": "error"},
    {"text": "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01", "coarse": "image", "modality": "binary_image"},
    {"text": "GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", "coarse": "image", "modality": "binary_image"},
    {"text": "PK\x03\x04\x14\x00\x08\x00\x08\x00\x00\x00!$\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", "coarse": "binary", "modality": "binary_archive"},
]

HEAD_NAMES = ["coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime"]
LABEL_MAP = {
    "coarse": {i: n for i, n in enumerate(COARSE_LABELS)},
    "modality": {i: n for i, n in enumerate(MODALITY_LABELS)},
    "subtype": {i: n for i, n in enumerate(SUBTYPE_LABELS)},
    "code_lang": {i: n for i, n in enumerate(CODE_LANG_LABELS)},
    "text_lang": {i: n for i, n in enumerate(TEXT_LANG_LABELS)},
    "file_mime": {i: n for i, n in enumerate(FILE_MIME_LABELS)},
}


def _eval_real_world(model, device):
    model.eval()
    correct = 0
    total = len(REAL_WORLD_INPUTS)
    with torch.no_grad():
        for item in REAL_WORLD_INPUTS:
            text = item["text"]
            data = text.encode("utf-8")[:model.config.max_bytes]
            ids = torch.zeros(1, len(data), dtype=torch.long, device=device)
            ids[0, :len(data)] = torch.tensor(list(data), dtype=torch.long, device=device)
            mask = torch.ones(1, len(data), dtype=torch.bool, device=device)
            logits = model(ids, mask)
            ok = True
            for head in ["coarse", "modality", "subtype", "code_lang", "text_lang"]:
                if head in item and head in logits:
                    pred = LABEL_MAP[head][logits[head].argmax(dim=-1).item()]
                    if pred != item[head]:
                        ok = False
            if ok:
                correct += 1
    return correct / total, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="checkpoints/best.pt")
    parser.add_argument("--data", default="checkpoints/real_samples.pkl")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--coarse-weight", type=float, default=8.0)
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  LR: {args.lr}  Steps: {args.steps}  Coarse weight: {args.coarse_weight}")

    # Load checkpoint
    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.train()
    print(f"Loaded {args.resume}")

    # Baseline eval
    acc_before, n_correct, n_total = _eval_real_world(model, device)
    print(f"Before: {acc_before*100:.1f}% ({n_correct}/{n_total})")

    # Load real samples
    with open(args.data, "rb") as f:
        data = pickle.load(f)
        real_samples = data.get("samples", data) if isinstance(data, dict) else data
    print(f"Real samples: {len(real_samples)}")

    # Setup — use same head weights as original training (coarse=8.0)
    head_weights = {
        "coarse": args.coarse_weight,
        "modality": 2.0,
        "subtype": 1.0,
        "code_lang": 1.5,
        "text_lang": 1.5,
        "file_mime": 1.0,
        "risk": 1.0,
    }
    criterion = MultiTaskLoss(head_weights)

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "linears" not in n], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if "linears" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98))

    gen = SyntheticGenerator(seed=42)
    step = 0
    best_acc = acc_before
    warmup_steps = 100

    while step < args.steps:
        # Synthetic batch
        synth_batch = [gen() for _ in range(16)]
        collated = collate_fn(synth_batch)
        ids = collated["input_ids"].to(device)
        mask = collated["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in collated["labels"].items()}

        # Real batch
        real_bs = int(16 * args.real_ratio)
        real_idx = (step * real_bs) % max(1, len(real_samples))
        real_batch = real_samples[real_idx:real_idx + real_bs]
        if len(real_batch) < real_bs:
            real_batch = real_samples[:real_bs]

        # LR warmup
        if step < warmup_steps:
            lr = args.lr * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, ("base",))

        if real_batch:
            real_collated = collate_fn(real_batch)
            real_ids = real_collated["input_ids"].to(device)
            real_mask = real_collated["attention_mask"].to(device)
            real_labels = {k: v.to(device) for k, v in real_collated["labels"].items()}
            r_loss, r_comps = multi_tier_loss(model, real_ids, real_mask, real_labels, criterion, ("base",))
            loss = loss + r_loss
            for k, v in r_comps.items():
                comps[f"r_{k}"] = v

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            parts = [f"step={step:5d} lr={lr:.7f} loss={loss.item():.4f}"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            if any(k.startswith("r_") for k in comps):
                r_vals = {k: comps[k] for k in comps if k.startswith("r_")}
                parts.append(f"real={sum(r_vals.values()):.2f}")
            print("  ".join(parts))

        step += 1

        # Eval
        if step % 500 == 0 or step == args.steps:
            acc, n_c, n_t = _eval_real_world(model, device)
            print(f"  EVAL step={step}: real_world={acc*100:.1f}% ({n_c}/{n_t})")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "real_world_accuracy": acc,
                    "real_world_correct": n_c,
                    "real_world_total": n_t,
                }, os.path.join(args.output, "finetuned_best.pt"))
                print(f"  New best! saved to {args.output}/finetuned_best.pt")
            model.train()

    # Final eval
    acc, n_c, n_t = _eval_real_world(model, device)
    print(f"\nFinal: {acc*100:.1f}% ({n_c}/{n_t})")
    print(f"Baseline was: {acc_before*100:.1f}%")
    delta = (acc - acc_before) * 100
    print(f"Delta: {delta:+.1f}%")

    if acc > acc_before:
        print("\n✓ Improved! Replacing checkpoint...")
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "real_world_accuracy": acc,
        }, os.path.join(args.output, "best.pt"))
        print(f"Replaced {args.output}/best.pt")
    else:
        print(f"\n✗ Did not improve. Best was {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()
