"""Round 7: Fix French text detected as Spanish."""

from __future__ import annotations

import math
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.data import SyntheticGenerator, Sample
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


def gen_fr_samples(rng=random.Random(123)):
    samples = []
    # French phrases with distinctive vocabulary
    fr_phrases = [
        "Bonjour le monde! Comment allez-vous aujourd'hui?",
        "Je m'appelle Pierre et j'habite a Paris depuis cinq ans.",
        "Le chat est sur la table et le chien dort dans le jardin.",
        "Quel temps magnifique aujourd'hui! Il fait beau et chaud.",
        "Je voudrais un cafe noir s'il vous plait avec du sucre.",
        "L'intelligence artificielle transforme notre monde rapidement.",
        "C'est une excellente idee, merci beaucoup pour votre aide!",
        "La vie est belle quand on sait apprecier les petites choses.",
        "Pouvez-vous m'aider a traduire ce document en francais?",
        "Nous devons resoudre ce probleme ensemble aujourd'hui.",
        "Ou se trouve la bibliotheque la plus proche du centre-ville?",
        "Le developpement web est un domaine passionnant et creatif.",
        "Je suis etudiant a l'universite de Strasbourg en France.",
        "La Tour Eiffel est le monument le plus visite de France.",
        "Nous allons au cinema ce soir pour voir un nouveau film.",
    ]
    # Spanish distractors
    es_phrases = [
        "Hola mundo! Como estas hoy? Espero que todo este bien.",
        "Me llamo Juan y vivo en Madrid, Espana.",
        "El gato esta en la mesa y el perro duerme en el jardin.",
        "Que tiempo magnifico hoy! Hace sol y calor.",
        "Quisiera un cafe negro por favor.",
        "La inteligencia artificial transforma nuestro mundo rapidamente.",
        "Es una excelente idea, muchas gracias por tu ayuda!",
        "La vida es bella cuando sabes apreciar las cosas pequenas.",
    ]

    for _ in range(2000):
        text = rng.choice(fr_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))

    for _ in range(200):
        text = rng.choice(es_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["es"],
        ))

    rng.shuffle(samples)
    return samples


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    acc_before, _, _ = _eval_real_world(model, _DEVICE)
    print(f"Before: {acc_before*100:.1f}%")

    real_samples = []
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            d = pickle.load(f)
            real_samples = d.get("samples", d) if isinstance(d, dict) else d

    fr_samples = gen_fr_samples()
    print(f"FR samples: {len(fr_samples)}, Real: {len(real_samples)}")

    gen_synth = SyntheticGenerator(seed=42)
    hw = {
        "coarse": 8.0, "modality": 2.0, "subtype": 1.0,
        "code_lang": 1.5, "text_lang": 3.0,
        "file_mime": 1.0, "risk": 1.0,
    }
    criterion = MultiTaskLoss(hw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=0.01)

    total_steps = 2000
    best_acc = acc_before
    best_state = model.state_dict()
    model.train()

    for step in range(total_steps):
        bs = 16
        n_fr = bs * 2 // 5
        n_real = bs * 3 // 10
        n_synth = bs - n_fr - n_real

        batch = [fr_samples[(step * n_fr + i) % len(fr_samples)] for i in range(n_fr)]
        if n_real > 0 and real_samples:
            ri = (step * n_real) % len(real_samples)
            batch.extend(real_samples[ri:ri + n_real])
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

    if final_acc >= acc_before:
        torch.save(
            {"model_state_dict": best_state, "step": total_steps,
             "config": model.config.__dict__, "real_world_accuracy": final_acc},
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")
    else:
        print("No improvement.")


if __name__ == "__main__":
    main()
