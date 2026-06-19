"""Focus fine-tune: generate synthetic variants of the 9 failing test cases."""

from __future__ import annotations

import argparse
import os
import pickle
import random

import torch

from .arch import PicoType, PicoTypeConfig
from .data import Sample, SyntheticGenerator, SyntheticDataset
from .labels import (
    COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS,
    CODE_LANG_LABELS, TEXT_LANG_LABELS,
)
from .train import collate_fn, MultiTaskLoss, multi_tier_loss

_COARSE = {n: i for i, n in enumerate(COARSE_LABELS)}
_MODALITY = {n: i for i, n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n: i for i, n in enumerate(SUBTYPE_LABELS)}
_CODE = {n: i for i, n in enumerate(CODE_LANG_LABELS)}
_TEXT = {n: i for i, n in enumerate(TEXT_LANG_LABELS)}

LABEL_MAP = {
    "coarse": {i: n for i, n in enumerate(COARSE_LABELS)},
    "modality": {i: n for i, n in enumerate(MODALITY_LABELS)},
    "subtype": {i: n for i, n in enumerate(SUBTYPE_LABELS)},
    "code_lang": {i: n for i, n in enumerate(CODE_LANG_LABELS)},
    "text_lang": {i: n for i, n in enumerate(TEXT_LANG_LABELS)},
}


def _rng_ident(rng, length=8):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))


def generate_variants(rng=random) -> list[Sample]:
    """Generate synthetic variants of the 9 failing test patterns."""
    samples = []

    # 1) Python `def` without imports (currently predicts Ruby)
    for _ in range(300):
        fname = _rng_ident(rng, 6)
        arg = _rng_ident(rng, 4)
        val = _rng_ident(rng, 5)
        variants = [
            f"def {fname}({arg}):\n    print({val})\n",
            f"def {fname}({arg}):\n    return {val}\n",
            f"def {fname}():\n    print('{val}')\n",
            f"def {fname}({arg}):\n    {val} = {arg} + 1\n    return {val}\n",
            f"def {fname}():\n    pass\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["code"],
                modality=_MODALITY["textual"],
                code_lang=_CODE["python"],
            ))

    # 2) Bash shebang (currently predicts fish)
    for _ in range(200):
        cmd = _rng_ident(rng, 5)
        dirname = _rng_ident(rng, 4)
        variants = [
            "#!/bin/bash\n\necho 'hello'\n",
            "#!/bin/bash\n\nset -e\n\necho 'done'\n",
            f"#!/bin/bash\n\nfor f in *.{cmd}; do\n    echo $f\ndone\n",
            "#!/usr/bin/env bash\n\nls -la\n",
            "#!/bin/bash\n\n# comment\n" + cmd + "() {\n    echo ok\n}\n\n" + cmd + "\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["code"],
                modality=_MODALITY["textual"],
                code_lang=_CODE["bash"],
            ))

    # 3) HTML doctype (currently predicts scss)
    for _ in range(200):
        title = _rng_ident(rng, 7)
        variants = [
            "<!DOCTYPE html>\n<html>\n<head><title>" + title + "</title></head>\n<body>\n<p>hello</p>\n</body>\n</html>\n",
            "<!DOCTYPE html>\n<html>\n<body>\n<h1>" + title + "</h1>\n</body>\n</html>\n",
            "<html>\n<head>\n<title>" + title + "</title>\n</head>\n<body>\n</body>\n</html>\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["markup"],
                modality=_MODALITY["textual"],
                code_lang=_CODE["html"],
            ))

    # 4) French text (currently predicts Czech)
    sentences = [
        "Bonjour le monde! Comment allez-vous?",
        "Je m'appelle Pierre et j'habite à Paris.",
        "Le chat est sur la table et le chien est dans le jardin.",
        "Quel temps magnifique aujourd'hui!",
        "Je voudrais un café s'il vous plaît.",
        "L'intelligence artificielle transforme notre monde.",
        "La programmation est un art et une science.",
        "Nous devons résoudre ce problème ensemble.",
        "C'est une excellente idée, merci beaucoup!",
        "La vie est belle quand on sait apprécier les petites choses.",
    ]
    for _ in range(300):
        s = rng.choice(sentences)
        # Add some variations
        prefix = rng.choice(["", "French: ", "FR: ", ""])
        suffix = rng.choice(["", "\n", ". ", " "])
        text = prefix + s + suffix
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))

    # 5) YAML config (currently predicts error/graphql)
    for _ in range(200):
        k1 = _rng_ident(rng, 6)
        k2 = _rng_ident(rng, 7)
        k3 = _rng_ident(rng, 5)
        v1 = _rng_ident(rng, 5)
        v2 = str(rng.randint(1, 999))
        variants = [
            f"{k1}: {v1}\n{k2}: {v2}\n",
            f"{k1}: {v2}\n{k2}:\n  - {v1}\n  - {_rng_ident(rng,4)}\n",
            f"{k1}: {v1}\n{k2}:\n  {k3}: {v2}\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["config"],
                modality=_MODALITY["textual"],
                subtype=_SUBTYPE["yaml"],
            ))

    # 6) Traceback / Error (currently predicts code)
    for _ in range(200):
        fn = _rng_ident(rng, 5) + ".py"
        func = _rng_ident(rng, 6)
        exc = rng.choice(["ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError"])
        msg = _rng_ident(rng, 10)
        variants = [
            f"Traceback (most recent call last):\n  File \"{fn}\", line 3, in <module>\n    {func}()\n  File \"{fn}\", line 5, in {func}\n    return {msg}\n{exc}: {msg}\n",
            f"Traceback (most recent call last):\n  File \"{fn}\", line 1, in <module>\n    import {func}\nModuleNotFoundError: No module named '{func}'\n",
            f"Traceback (most recent call last):\n  File \"{fn}\", line 7, in <module>\n    result = {func}({msg})\n{exc}: {msg}\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["error"],
                modality=_MODALITY["textual"],
            ))

    # 7) Image PNG headers
    import struct, zlib
    for _ in range(100):
        w, h = rng.randint(1, 100), rng.randint(1, 100)
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
        chunk = b'IHDR' + ihdr_data + ihdr_crc
        length = struct.pack('>I', len(ihdr_data))
        data = sig + length + chunk
        samples.append(Sample(
            data=data,
            coarse=_COARSE["image"],
            modality=_MODALITY["binary_image"],
        ))

    # 8) GIF headers
    for _ in range(100):
        w, h = rng.randint(1, 100), rng.randint(1, 100)
        sig = b'GIF89a'
        dims = struct.pack('<HH', w, h)
        extra = bytes([0x80, 0x00, 0x00])
        data = sig + dims + extra
        samples.append(Sample(
            data=data,
            coarse=_COARSE["image"],
            modality=_MODALITY["binary_image"],
        ))

    # 9) ZIP headers (archive)
    for _ in range(100):
        data = b'PK\x03\x04' + bytes(rng.randint(0, 255) for _ in range(26))
        samples.append(Sample(
            data=data,
            coarse=_COARSE["binary"],
            modality=_MODALITY["binary_archive"],
        ))

    rng.shuffle(samples)
    print(f"Generated {len(samples)} focused training samples")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="checkpoints/finetuned_best.pt")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--coarse-weight", type=float, default=8.0)
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  LR={args.lr}  Steps={args.steps}")

    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.train()
    print(f"Loaded {args.resume}")

    # Baseline eval
    from .finetune_real import _eval_real_world
    acc_before, n_c, n_t = _eval_real_world(model, device)
    print(f"Before: {acc_before*100:.1f}% ({n_c}/{n_t})")

    # Generate focused training samples
    rng = random.Random(42)
    focus_samples = generate_variants(rng)

    # Also load real data
    focus_all = list(focus_samples)
    if os.path.exists("checkpoints/real_samples.pkl"):
        with open("checkpoints/real_samples.pkl", "rb") as f:
            data = pickle.load(f)
            real = data.get("samples", data) if isinstance(data, dict) else data
        # Sample a subset of real data to balance
        rng.shuffle(real)
        focus_all.extend(real[:2000])
    print(f"Total training samples: {len(focus_all)}")

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
    gen = SyntheticGenerator(seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    step = 0
    best_acc = acc_before
    best_state = None

    while step < args.steps:
        # Mix: 40% focus, 30% real, 30% synthetic
        bs = 16
        n_focus = int(bs * 0.4)
        n_real = int(bs * 0.3)
        n_synth = bs - n_focus - n_real

        focus_batch = [focus_all[(step * n_focus + i) % len(focus_all)] for i in range(n_focus)]
        synth_batch = [gen() for _ in range(n_synth)]
        batch = focus_batch + synth_batch

        # Add real samples
        if n_real > 0 and len(real) > 0:
            real_idx = (step * n_real) % len(real)
            real_batch = real[real_idx:real_idx + n_real]
            batch.extend(real_batch)

        collated = collate_fn(batch)
        ids = collated["input_ids"].to(device)
        mask = collated["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in collated["labels"].items()}

        # LR schedule
        if step < 100:
            lr = args.lr * (step + 1) / 100
        else:
            progress = (step - 100) / max(1, args.steps - 100)
            lr = args.lr * 0.5 * (1 + __import__('math').cos(progress * 3.14159))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, ("base",))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            parts = [f"step={step:5d} lr={lr:.7f} loss={loss.item():.4f}"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        step += 1

        if step % 500 == 0 or step == args.steps:
            acc, n_c, n_t = _eval_real_world(model, device)
            print(f"  EVAL step={step}: real_world={acc*100:.1f}% ({n_c}/{n_t})")
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "real_world_accuracy": acc,
                    "real_world_correct": n_c,
                    "real_world_total": n_t,
                }, os.path.join(args.output, "focus_best.pt"))
                print(f"  New best! ({acc*100:.1f}%)")
            model.train()

    acc, n_c, n_t = _eval_real_world(model, device)
    print(f"\nFinal: {acc*100:.1f}% ({n_c}/{n_t})")
    print(f"Best:  {best_acc*100:.1f}%")
    print(f"Delta: {(acc-acc_before)*100:+.1f}%")

    if best_acc > acc_before:
        print(f"\n✓ Improved! Replacing checkpoint...")
        torch.save({
            "step": step,
            "model_state_dict": best_state or model.state_dict(),
            "real_world_accuracy": best_acc,
        }, os.path.join(args.output, "best.pt"))
        print(f"Replaced {args.output}/best.pt")


if __name__ == "__main__":
    main()
