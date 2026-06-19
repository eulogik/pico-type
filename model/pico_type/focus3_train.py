"""Round 3: fix last 3 failures (French, YAML coarse, Traceback coarse)."""

from __future__ import annotations

import argparse
import os
import pickle
import random

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


def gen(rng=random.Random(7)):
    samples = []

    # 1) French text (currently predicts Spanish "es")
    fr_sentences = [
        "Bonjour le monde! Comment allez-vous aujourd'hui?",
        "Je m'appelle Pierre et j'habite à Paris depuis cinq ans.",
        "Le chat est sur la table et le chien dort dans le jardin vert.",
        "Quel temps magnifique aujourd'hui! Il fait beau et chaud.",
        "Je voudrais un café noir s'il vous plaît avec du sucre.",
        "L'intelligence artificielle transforme notre monde rapidement.",
        "La programmation est un art et une science à la fois passionnante.",
        "Nous devons résoudre ce problème ensemble aujourd'hui même.",
        "C'est une excellente idée, merci beaucoup pour votre aide précieuse!",
        "La vie est belle quand on sait apprécier les petites choses simples.",
        "Je suis heureux de vous rencontrer enfin aujourd'hui.",
        "Où se trouve la bibliothèque la plus proche du centre-ville?",
        "Pouvez-vous m'aider à traduire ce document en français?",
        "Le développement web est un domaine passionnant et créatif.",
        "Les données sont analysées par le système automatiquement chaque jour.",
        "Il faut que nous parlions de ce projet important demain.",
        "J'ai besoin d'un nouveau ordinateur portable pour travailler.",
        "La nouvelle application fonctionne parfaitement sur mon téléphone.",
        "Nous avons terminé le projet avant la date limite prévue.",
        "C'est un véritable plaisir de travailler avec une équipe talentueuse.",
        "Je pense que la solution la plus simple est souvent la meilleure.",
        "Le musée du Louvre est l'un des plus grands musées du monde.",
        "La tour Eiffel brille de mille feux chaque soir à Paris.",
        "Pouvez-vous répéter s'il vous plaît? Je n'ai pas bien entendu.",
        "Merci infiniment pour votre gentillesse et votre générosité!",
    ]
    for _ in range(800):
        s = rng.choice(fr_sentences)
        if rng.random() < 0.3:
            s += " " + rng.choice(fr_sentences).split(".")[0]
        samples.append(Sample(
            data=s.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))

    # Also add Spanish samples labeled correctly so model learns the difference
    es_sentences = [
        "Hola mundo! Cómo estás hoy?",
        "Me llamo Juan y vivo en Madrid.",
        "El gato está en la mesa y el perro duerme en el jardín.",
        "Qué tiempo magnífico hoy! Hace sol y calor.",
        "Quisiera un café negro por favor.",
        "La inteligencia artificial transforma nuestro mundo rápidamente.",
        "La programación es un arte y una ciencia fascinante.",
        "Tenemos que resolver este problema juntos hoy mismo.",
        "Es una excelente idea, muchas gracias por tu valiosa ayuda!",
        "La vida es bella cuando sabes apreciar las pequeñas cosas.",
        "Estoy feliz de conocerte finalmente hoy.",
        "Dónde está la biblioteca más cercana al centro?",
        "Puedes ayudarme a traducir este documento al español?",
        "El desarrollo web es un campo apasionante y creativo.",
        "Los datos son analizados por el sistema automáticamente cada día.",
    ]
    for _ in range(100):
        s = rng.choice(es_sentences)
        samples.append(Sample(
            data=s.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["es"],
        ))

    # 2) YAML config → coarse=error (should be config; subtype already correct)
    for _ in range(600):
        k1 = _id(rng, 7)
        k2 = _id(rng, 5)
        v1 = _id(rng, 6)
        v2 = str(rng.randint(0, 999))
        v3 = _id(rng, 5)
        variants = [
            f"{k1}: {v1}\n{k2}: {v2}\n",
            f"{k1}:\n  - {v3}\n  - {v1}\n{k2}: {v2}\n",
            f"{k1}:\n  {k2}: {v2}\n  {_id(rng,4)}: {v3}\n",
            f"# config file\n{k1}: {v1}\n{k2}: {v2}\n",
            f"{k1}:\n  {_id(rng,6)}:\n    - {v3}\n    - {v2}\n",
            f"{k1}: {v1}\n{k2}:\n  {_id(rng,5)}: {v3}\n  {_id(rng,4)}: true\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["config"],
                modality=_MODALITY["textual"],
                subtype=_SUBTYPE["yaml"],
            ))

    # 3) Traceback → coarse=code (should be error)
    for _ in range(500):
        fn = _id(rng, 5) + ".py"
        func = _id(rng, 6)
        inner = _id(rng, 5)
        exc = rng.choice(["ValueError", "TypeError", "KeyError", "IndexError",
                          "ZeroDivisionError", "AttributeError", "ImportError",
                          "RuntimeError", "FileNotFoundError", "PermissionError"])
        msg = _id(rng, 10)
        variants = [
            f"Traceback (most recent call last):\n  File \"{fn}\", line 3, in <module>\n    {func}()\n  File \"{fn}\", line 5, in {func}\n    return {msg}\n{exc}: {msg}\n",
            f"Traceback (most recent call last):\n  File \"{fn}\", line 1, in <module>\n    import {func}\nModuleNotFoundError: No module named '{func}'\n",
            f"Traceback (most recent call last):\n  File \"{fn}\", line 7, in <module>\n    result = {func}({msg})\n{exc}: {msg}\n",
            f"Traceback (most recent call last):\n  File \"{fn}\", line 10, in {func}\n    {inner} = {func}({msg})\n  File \"{fn}\", line 5, in {func}\n    if {msg} is None:\n{exc}: {msg} is not allowed\n",
            f"Traceback (most recent call last):\n  File \"{fn}\", line 2, in <module>\n    {func}()\n  File \"/usr/lib/python3.11/{func}.py\", line 42, in {func}\n    raise {exc}(\"{msg}\")\n{exc}: {msg}\n",
        ]
        for v in variants:
            samples.append(Sample(
                data=v.encode("utf-8"),
                coarse=_COARSE["error"],
                modality=_MODALITY["textual"],
            ))

    rng.shuffle(samples)
    print(f"Round 3: {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="checkpoints/best.pt")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  LR={args.lr}  Steps={args.steps}")

    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.train()

    acc_before, n_c, n_t = _eval_real_world(model, device)
    print(f"Before: {acc_before*100:.1f}% ({n_c}/{n_t})")

    focus = gen()
    all_data = list(focus)
    if os.path.exists("checkpoints/real_samples.pkl"):
        with open("checkpoints/real_samples.pkl", "rb") as f:
            d = pickle.load(f)
            real = d.get("samples", d) if isinstance(d, dict) else d
            random.Random(7).shuffle(real)
            all_data.extend(real[:1500])
    print(f"Total: {len(all_data)} samples")

    head_weights = {"coarse": 8.0, "modality": 2.0, "subtype": 1.0,
                    "code_lang": 1.5, "text_lang": 1.5, "file_mime": 1.0, "risk": 1.0}
    criterion = MultiTaskLoss(head_weights)
    gen_synth = SyntheticGenerator(seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    step = 0
    best_acc = acc_before
    best_state = None

    while step < args.steps:
        bs = 16
        n_f = int(bs * 0.55)
        n_r = int(bs * 0.25)
        n_s = bs - n_f - n_r

        fb = [all_data[(step * n_f + i) % len(all_data)] for i in range(n_f)]
        if n_r > 0 and len(real) > 0:
            ri = (step * n_r) % len(real)
            fb.extend(real[ri:ri + n_r])
        fb.extend([gen_synth() for _ in range(n_s)])

        collated = collate_fn(fb)
        ids = collated["input_ids"].to(device)
        mask = collated["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in collated["labels"].items()}

        lr = args.lr * min(1.0, (step + 1) / 100) if step < 100 else args.lr * 0.5 * (1 + __import__('math').cos((step - 100) / max(1, args.steps - 100) * 3.14159))
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
                print("  New best!")
            model.train()

    if best_state is not None and best_acc > acc_before:
        torch.save({"step": step, "model_state_dict": best_state, "real_world_accuracy": best_acc},
                    os.path.join(args.output, "best.pt"))
        print(f"✓ Replaced best.pt ({best_acc*100:.1f}% > {acc_before*100:.1f}%)")


if __name__ == "__main__":
    main()
