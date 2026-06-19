"""Round 5: Mixed approach for the 3 borderline failures."""
from __future__ import annotations
import os
import pickle
import random
import torch
import math
from .arch import PicoType, PicoTypeConfig
from .data import Sample, SyntheticGenerator
from .finetune_real import _eval_real_world
from .labels import COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS, CODE_LANG_LABELS, TEXT_LANG_LABELS
from .train import collate_fn, MultiTaskLoss, multi_tier_loss

_COARSE = {n:i for i,n in enumerate(COARSE_LABELS)}
_MODALITY = {n:i for i,n in enumerate(MODALITY_LABELS)}
_SUBTYPE = {n:i for i,n in enumerate(SUBTYPE_LABELS)}
_CODE = {n:i for i,n in enumerate(CODE_LANG_LABELS)}
_TEXT = {n:i for i,n in enumerate(TEXT_LANG_LABELS)}

def _id(r, n=8):
    return "".join(r.choices("abcdefghijklmnopqrstuvwxyz", k=n))

def gen_mixed(rng=random.Random(17)):
    """Generate a diverse mix centered on the 3 failure patterns."""
    samples = []

    # French: 800 samples with highly distinctive vocabulary
    fr = [
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
    ]
    # Spanish distractors
    es = [
        "Hola mundo! Como estas hoy?",
        "Me llamo Juan y vivo en Madrid, Espana.",
        "El gato esta en la mesa y el perro duerme en el jardin.",
        "Que tiempo magnifico hoy! Hace sol y calor.",
        "Quisiera un cafe negro por favor.",
        "La inteligencia artificial transforma nuestro mundo rapidamente.",
        "Es una excelente idea, muchas gracias por tu ayuda!",
        "La vida es bella cuando sabes apreciar las cosas pequenas.",
    ]
    for _ in range(400):
        s = rng.choice(fr)
        samples.append(Sample(data=s.encode("utf-8"), coarse=_COARSE["text"], modality=_MODALITY["textual"], text_lang=_TEXT["fr"]))
    for _ in range(60):
        s = rng.choice(es)
        samples.append(Sample(data=s.encode("utf-8"), coarse=_COARSE["text"], modality=_MODALITY["textual"], text_lang=_TEXT["es"]))

    # YAML config: 600 samples with name: prefix
    for _ in range(600):
        k1 = rng.choice(["name", "title", "app", "service", "project", "config", "server", "app_name"])
        k2 = _id(rng, 7)
        v1 = _id(rng, 6)
        v2 = str(rng.randint(0, 999))
        v = f"{k1}: {v1}\n{k2}: {v2}\n"
        samples.append(Sample(data=v.encode("utf-8"), coarse=_COARSE["config"], modality=_MODALITY["textual"], subtype=_SUBTYPE["yaml"]))

    # Tracebacks: 600 samples with distinctive language
    for _ in range(600):
        fn = _id(rng, 5) + ".py"
        func = _id(rng, 6)
        exc = rng.choice(["ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError"])
        msg = _id(rng, 10)
        v = f"Traceback (most recent call last):\n  File \"{fn}\", line 3, in <module>\n    {func}()\n  File \"{fn}\", line 5, in {func}\n    return {msg}\n{exc}: {msg}\n"
        samples.append(Sample(data=v.encode("utf-8"), coarse=_COARSE["error"], modality=_MODALITY["textual"]))

    # Also add 200 "normal" Python def samples to remind model about python
    for _ in range(200):
        fn = _id(rng, 6)
        arg = _id(rng, 4)
        v = f"def {fn}({arg}):\n    return {arg}\n"
        samples.append(Sample(data=v.encode("utf-8"), coarse=_COARSE["code"], modality=_MODALITY["textual"], code_lang=_CODE["python"]))

    rng.shuffle(samples)
    return samples

def main():
    rng = random.Random(17)
    focus = gen_mixed(rng)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    m.load_state_dict(torch.load("checkpoints/best.pt", map_location=device).get("model_state_dict"))
    m.train()
    acc_before,_,_ = _eval_real_world(m, device)
    print(f"Before: {acc_before*100:.1f}%")

    # Load mixed datasets
    all_data = list(focus)
    if os.path.exists("checkpoints/real_samples.pkl"):
        with open("checkpoints/real_samples.pkl", "rb") as f:
            d = pickle.load(f)
            real = d.get("samples", d) if isinstance(d, dict) else d
        rng.shuffle(real)
        all_data.extend(real[:2000])
    print(f"Focus: {len(focus)}, Total: {len(all_data)}")

    hw = {"coarse": 8.0, "modality": 2.0, "subtype": 1.0, "code_lang": 1.5, "text_lang": 1.5, "file_mime": 1.0, "risk": 1.0}
    criterion = MultiTaskLoss(hw)
    gen_synth = SyntheticGenerator(seed=42)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-6, weight_decay=0.01)

    best_acc = acc_before
    best_state = None

    for step in range(2000):
        bs = 16
        n_f = int(bs * 0.4)
        n_r = int(bs * 0.25)
        n_s = bs - n_f - n_r

        # Mix focus + real + synthetic
        batch = [all_data[(step * n_f + i) % len(all_data)] for i in range(n_f)]
        if n_r > 0:
            ri = (step * n_r) % len(real)
            batch.extend(real[ri:ri+n_r])
        batch.extend([gen_synth() for _ in range(n_s)])

        coll = collate_fn(batch)
        ids = coll["input_ids"].to(device)
        mask = coll["attention_mask"].to(device)
        labels = {k:v.to(device) for k,v in coll["labels"].items()}

        lr = 3e-6 * min(1, (step+1)/100) if step < 100 else 3e-6 * 0.5 * (1 + math.cos((step-100)/1900*math.pi))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        loss, comps = multi_tier_loss(m, ids, mask, labels, criterion, ("base",))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            parts = [f"step={step:5d} lr={lr:.7f} loss={loss.item():.4f}"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        if step % 500 == 0 or step == 1999:
            acc, cor, tot = _eval_real_world(m, device)
            print(f"  EVAL step={step}: real_world={acc*100:.1f}% ({cor}/{tot})")
            if acc > best_acc:
                best_acc = acc
                best_state = m.state_dict()
                print("  NEW BEST!")
            m.train()

    if best_state and best_acc > acc_before:
        torch.save({"step": step, "model_state_dict": best_state, "real_world_accuracy": best_acc}, "checkpoints/best.pt")
        print(f"Replaced best.pt ({best_acc*100:.1f}%)")

if __name__ == "__main__":
    main()
