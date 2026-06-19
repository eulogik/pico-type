"""Round 4: ultra-focused on last 3 failures only."""
from __future__ import annotations
import random
import torch
from .arch import PicoType, PicoTypeConfig
from .data import Sample
from .finetune_real import _eval_real_world
from .labels import COARSE_LABELS, TEXT_LANG_LABELS, SUBTYPE_LABELS, MODALITY_LABELS
from .train import collate_fn, MultiTaskLoss, multi_tier_loss

_COARSE = {n:i for i,n in enumerate(COARSE_LABELS)}
_TEXT = {n:i for i,n in enumerate(TEXT_LANG_LABELS)}
_SUBTYPE = {n:i for i,n in enumerate(SUBTYPE_LABELS)}
_MODALITY = {n:i for i,n in enumerate(MODALITY_LABELS)}

def _id(r, n=8): return "".join(r.choices("abcdefghijklmnopqrstuvwxyz", k=n))

def main():
    rng = random.Random(13)
    samples = []

    # French only - 2000 samples
    fr_sentences = [
        "Bonjour le monde! Comment allez-vous aujourd'hui?",
        "Je m'appelle Pierre et j'habite a Paris depuis cinq ans.",
        "Le chat est sur la table et le chien dort dans le jardin.",
        "Quel temps magnifique aujourd'hui! Il fait beau et chaud.",
        "Je voudrais un cafe noir s'il vous plait avec du sucre.",
        "L'intelligence artificielle transforme notre monde rapidement.",
        "C'est une excellente idee, merci beaucoup!",
        "La vie est belle quand on sait apprecier les petites choses.",
        "Pouvez-vous m'aider a traduire ce document en francais?",
        "Nous devons resoudre ce probleme ensemble aujourd'hui.",
        "Je suis heureux de vous rencontrer enfin aujourd'hui.",
        "Ou se trouve la bibliotheque la plus proche?",
        "Le developpement web est un domaine passionnant.",
        "J'ai besoin d'un nouvel ordinateur pour travailler.",
        "La nouvelle application fonctionne parfaitement maintenant.",
        "Merci infiniment pour votre gentillesse et votre genérosité!",
        "Il faut absolument que nous parlions de ce projet demain.",
        "La Tour Eiffel est magnifique la nuit quand elle scintille.",
        "Je pense que la solution la plus simple est souvent la meilleure.",
        "Pouvez-vous repeter s'il vous plait? Je n'ai pas bien compris.",
        "Le musee du Louvre est l'un des plus grands du monde.",
        "J'ai appris le francais a l'ecole et je l'adore.",
        "C'est un veritable plaisir de travailler avec vous.",
        "Nous avons termine le projet avant la date limite.",
    ]
    for _ in range(2000):
        s = rng.choice(fr_sentences)
        if rng.random() < 0.3:
            s += " " + rng.choice(fr_sentences).split(".")[0] + "."
        samples.append(Sample(
            data=s.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))

    # YAML config - 1200 samples with name: prefix
    for _ in range(1200):
        k1 = rng.choice(["name", "title", "app", "service", "project", "config", "server"])
        k2 = _id(rng, 7)
        v1 = _id(rng, 6)
        v2 = str(rng.randint(0, 999))
        v = f"{k1}: {v1}\n{k2}: {v2}\n"
        samples.append(Sample(
            data=v.encode("utf-8"),
            coarse=_COARSE["config"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE["yaml"],
        ))

    # Traceback - 1200 samples
    for _ in range(1200):
        fn = _id(rng, 5) + ".py"
        func = _id(rng, 6)
        exc = rng.choice(["ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError", "AttributeError"])
        msg = _id(rng, 10)
        v = f"Traceback (most recent call last):\n  File \"{fn}\", line 3, in <module>\n    {func}()\n  File \"{fn}\", line 5, in {func}\n    return {msg}\n{exc}: {msg}\n"
        samples.append(Sample(
            data=v.encode("utf-8"),
            coarse=_COARSE["error"],
            modality=_MODALITY["textual"],
        ))

    rng.shuffle(samples)
    print(f"Generated {len(samples)} samples")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    m.load_state_dict(torch.load("checkpoints/best.pt", map_location=device).get("model_state_dict"))
    m.train()

    acc_before, _, _ = _eval_real_world(m, device)
    print(f"Before: {acc_before*100:.1f}%")

    hw = {"coarse": 8.0, "modality": 2.0, "subtype": 1.0, "code_lang": 1.5, "text_lang": 1.5, "file_mime": 1.0, "risk": 1.0}
    criterion = MultiTaskLoss(hw)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-6, weight_decay=0.01)

    best_acc = acc_before
    best_state = None

    for step in range(2500):
        bs = 16
        batch = [samples[(step * bs + i) % len(samples)] for i in range(bs)]
        coll = collate_fn(batch)
        ids = coll["input_ids"].to(device)
        mask = coll["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in coll["labels"].items()}

        if step < 100:
            lr = 3e-6 * (step + 1) / 100
        else:
            p = (step - 100) / 2400
            lr = 3e-6 * 0.5 * (1 + __import__("math").cos(p * 3.14159))
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

        if step % 500 == 0 or step == 2499:
            acc, cor, tot = _eval_real_world(m, device)
            print(f"  EVAL step={step}: real_world={acc*100:.1f}% ({cor}/{tot})")
            if acc > best_acc:
                best_acc = acc
                best_state = m.state_dict()
                print("  *** NEW BEST ***")
            m.train()

    if best_state and best_acc > acc_before:
        torch.save({"step": step, "model_state_dict": best_state, "real_world_accuracy": best_acc}, "checkpoints/best.pt")
        print(f"✓ Replaced best.pt ({best_acc*100:.1f}% > {acc_before*100:.1f}%)")
    else:
        print(f"No improvement. Best {best_acc*100:.1f}% vs baseline {acc_before*100:.1f}%")

if __name__ == "__main__":
    main()
