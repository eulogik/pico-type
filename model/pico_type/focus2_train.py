"""Round 2 focus training: fix remaining 6 failures."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import struct
import zlib

import torch

from .arch import PicoType, PicoTypeConfig
from .data import Sample, SyntheticGenerator
from .finetune_real import _eval_real_world
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


def _id(rng, n=8):
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n))


def gen_focus2(rng=random.Random(42)):
    samples = []

    # 1) Go package main (currently predicts kotlin)
    for _ in range(400):
        fn = _id(rng, 6)
        pkg = rng.choice(["main", _id(rng,5)])
        variants = [
            f"package {pkg}\n\nimport \"fmt\"\n\nfunc {fn}() {{\n    fmt.Println(\"{_id(rng,8)}\")\n}}\n",
            f"package {pkg}\n\nimport \"fmt\"\n\nfunc main() {{\n    fmt.Println(\"{_id(rng,8)}\")\n}}\n",
            f"package {pkg}\n\nimport (\n    \"fmt\"\n)\n\nfunc main() {{\n    fmt.Println(\"{_id(rng,8)}\")\n}}\n",
            f"package {pkg}\n\nimport \"fmt\"\n\nfunc {fn}(x int) int {{\n    return x + {rng.randint(1,99)}\n}}\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["code"],
                modality=_MODALITY["textual"],
                code_lang=_CODE["go"],
            ))

    # Also add Kotlin distractors with label kotlin to help model distinguish
    for _ in range(200):
        ns = _id(rng, 6)
        fn = _id(rng, 5)
        variants = [
            f"package {ns}\n\nfun {fn}() {{\n    println(\"{_id(rng,8)}\")\n}}\n",
            f"fun main() {{\n    println(\"{_id(rng,8)}\")\n}}\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["code"],
                modality=_MODALITY["textual"],
                code_lang=_CODE["kotlin"],
            ))

    # 2) French text (currently predicts Czech)
    import unicodedata
    fr_sentences = [
        "Bonjour le monde! Comment allez-vous aujourd'hui?",
        "Je m'appelle Pierre et j'habite à Paris depuis cinq ans.",
        "Le chat est sur la table et le chien dort dans le jardin.",
        "Quel temps magnifique aujourd'hui! Il fait beau et chaud.",
        "Je voudrais un café noir s'il vous plaît.",
        "L'intelligence artificielle transforme notre monde rapidement.",
        "La programmation est un art et une science à la fois.",
        "Nous devons résoudre ce problème ensemble aujourd'hui.",
        "C'est une excellente idée, merci beaucoup pour votre aide!",
        "La vie est belle quand on sait apprécier les petites choses.",
        "Je suis heureux de vous rencontrer aujourd'hui.",
        "Où se trouve la bibliothèque la plus proche?",
        "Pouvez-vous m'aider à traduire ce document?",
        "Le développement web est un domaine passionnant.",
        "Les données sont analysées par le système automatiquement.",
    ]
    cs_sentences = [
        "Dobrý den, jak se máte?",
        "Dnes je krásný den.",
        "Mám rád programování v Pythonu.",
    ]
    for _ in range(500):
        s = rng.choice(fr_sentences)
        for _ in range(rng.randint(0, 2)):
            s += " " + rng.choice(fr_sentences)[:20]
        samples.append(Sample(
            data=s.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))
    # Also add Czech samples labeled correctly to teach distinction
    for _ in range(50):
        s = rng.choice(cs_sentences)
        samples.append(Sample(
            data=s.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["cs"],
        ))

    # 3) YAML config coarse (currently predicts error, but subtype correct)
    for _ in range(300):
        k1 = _id(rng, 7)
        k2 = _id(rng, 5)
        v1 = _id(rng, 6)
        v2 = str(rng.randint(1, 999))
        v3 = _id(rng, 5)
        variants = [
            f"{k1}: {v1}\n{k2}: {v2}\n",
            f"{k1}:\n  - {v3}\n  - {v1}\n{k2}: {v2}\n",
            f"{k1}:\n  {k2}: {v2}\n  {_id(rng,4)}: {v3}\n",
            f"# config\n{k1}: {v1}\n{k2}: {v2}\n",
            f"{k1}: {v1}\n{k2}:\n  {_id(rng,5)}:\n    - {v3}\n    - {v2}\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["config"],
                modality=_MODALITY["textual"],
                subtype=_SUBTYPE["yaml"],
            ))

    # 4) PNG: modality=textual instead of binary_image
    for _ in range(200):
        w, h = rng.randint(1, 200), rng.randint(1, 200)
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
        chunk = b'IHDR' + ihdr_data + crc
        length = struct.pack('>I', len(ihdr_data))
        data = sig + length + chunk + bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 32)))
        samples.append(Sample(
            data=data,
            coarse=_COARSE["image"],
            modality=_MODALITY["binary_image"],
        ))

    # 5) GIF: coarse=binary (should be image), modality=binary_other (should be binary_image)
    for _ in range(200):
        w, h = rng.randint(1, 200), rng.randint(1, 200)
        ver = rng.choice([b'GIF87a', b'GIF89a'])
        dims = struct.pack('<HH', w, h)
        extra = bytes([rng.randint(0, 255) for _ in range(7)])
        data = ver + dims + extra + bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 32)))
        samples.append(Sample(
            data=data,
            coarse=_COARSE["image"],
            modality=_MODALITY["binary_image"],
        ))

    # 6) ZIP: modality=binary_other (should be binary_archive)
    for _ in range(200):
        data = b'PK\x03\x04' + bytes(rng.randint(0, 255) for _ in range(26)) + bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 64)))
        samples.append(Sample(
            data=data,
            coarse=_COARSE["binary"],
            modality=_MODALITY["binary_archive"],
        ))

    # Add more binary_archive distractors (not PK) to teach the distinction
    for _ in range(100):
        magic = rng.choice([b'\x1f\x8b', b'BZh', b'\xfd7zXZ', b'Rar!\x1a\x07\x00'])
        data = magic + bytes(rng.randint(0, 255) for _ in range(rng.randint(20, 60)))
        samples.append(Sample(
            data=data,
            coarse=_COARSE["binary"],
            modality=_MODALITY["binary_archive"],
        ))

    rng.shuffle(samples)
    print(f"Round 2: {len(samples)} samples generated")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="checkpoints/best.pt")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  LR={args.lr}  Steps={args.steps}")

    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.train()
    print(f"Loaded {args.resume}")

    acc_before, n_c, n_t = _eval_real_world(model, device)
    print(f"Before: {acc_before*100:.1f}% ({n_c}/{n_t})")

    rng = random.Random(99)
    focus = gen_focus2(rng)

    # Load real data too
    all_data = list(focus)
    if os.path.exists("checkpoints/real_samples.pkl"):
        with open("checkpoints/real_samples.pkl", "rb") as f:
            d = pickle.load(f)
            real = d.get("samples", d) if isinstance(d, dict) else d
        rng.shuffle(real)
        all_data.extend(real[:2000])
    print(f"Total: {len(all_data)} samples")

    head_weights = {
        "coarse": 8.0, "modality": 2.0, "subtype": 1.0,
        "code_lang": 1.5, "text_lang": 1.5, "file_mime": 1.0, "risk": 1.0,
    }
    criterion = MultiTaskLoss(head_weights)
    gen = SyntheticGenerator(seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    step = 0
    best_acc = acc_before
    best_state = None

    while step < args.steps:
        bs = 16
        n_focus = int(bs * 0.5)
        n_real = int(bs * 0.25)
        n_synth = bs - n_focus - n_real

        fb = [all_data[(step * n_focus + i) % len(all_data)] for i in range(n_focus)]
        sb = [gen() for _ in range(n_synth)]

        if n_real > 0 and len(real) > 0:
            ri = (step * n_real) % len(real)
            rb = list(real[ri:ri + n_real])
            fb.extend(rb)
        fb.extend(sb)

        collated = collate_fn(fb)
        ids = collated["input_ids"].to(device)
        mask = collated["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in collated["labels"].items()}

        if step < 100:
            lr = args.lr * (step + 1) / 100
        else:
            p = (step - 100) / max(1, args.steps - 100)
            lr = args.lr * 0.5 * (1 + __import__('math').cos(p * 3.14159))
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
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "real_world_accuracy": acc,
                }, os.path.join(args.output, "focus2_best.pt"))
                best_state = model.state_dict()
                print(f"  New best! ({acc*100:.1f}%)")
            model.train()

    acc, n_c, n_t = _eval_real_world(model, device)
    print(f"\nFinal: {acc*100:.1f}% ({n_c}/{n_t})  Best: {best_acc*100:.1f}%  Delta: {(acc-acc_before)*100:+.1f}%")
    if best_acc > acc_before and best_state is not None:
        torch.save({
            "step": step,
            "model_state_dict": best_state,
            "real_world_accuracy": best_acc,
        }, os.path.join(args.output, "best.pt"))
        print(f"✓ Replaced best.pt")


if __name__ == "__main__":
    main()
