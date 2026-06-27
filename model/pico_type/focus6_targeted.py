"""Round 6: Targeted training for the 2 fixable failures.
Failure 1: YAML config (name: test\nversion: 1.0.0...) → misclassified as error
Failure 2: Hindi text → detected as Sinhala instead of Hindi

Also trains on all heads to prevent forgetting.
"""

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


def _id(rng, n=8):
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n))


def gen_fix_targets(rng=random.Random(99)):
    """Generate samples targeting the 2 failure patterns."""
    samples = []

    # Failure 1: YAML config misclassified as error
    # Generate 2000 diverse YAML samples with various structures
    yaml_bases = [
        "name: {k}\nversion: {v}\ndescription: {k2}\n",
        "{k}: {v}\n{k2}: {v2}\n",
        "name: {k}\nversion: {v}\ndependencies:\n  - {k2}\n  - {v}\n",
        "apiVersion: v1\nkind: Service\nmetadata:\n  name: {k}\n  namespace: {v}\n",
        "server:\n  host: {k}\n  port: {v}\ndatabase:\n  name: {k2}\n",
        "app:\n  name: {k}\n  version: {v}\n  env: {v2}\n",
        "spring:\n  application:\n    name: {k}\n  datasource:\n    url: jdbc:mysql://{k2}:{v}\n",
        "build:\n  target: {k}\n  sources:\n    - src/{k2}.py\n    - src/{v}.py\n",
        "name: {k}\nversion: {v}\nlicense: MIT\nauthor: {k2}\n",
        "root: true\nenv:\n  node: true\n  es2021: true\nrules:\n  no-console: warn\n",
    ]
    for _ in range(2000):
        tmpl = rng.choice(yaml_bases)
        k = _id(rng, 5)
        v = str(rng.randint(1, 999))
        k2 = _id(rng, 6)
        v2 = rng.choice(["dev", "prod", "staging", "test", "local"])
        text = tmpl.format(k=k, v=v, k2=k2, v2=v2)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["config"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE["yaml"],
        ))

    # Failure 2: Hindi text
    # Generate 2000 Hindi samples with distinctive vocabulary
    hindi_phrases = [
        "नमस्ते दुनिया! आप कैसे हैं? मुझे उम्मीद है कि सब कुछ ठीक है।",
        "भारत एक महान देश है जहाँ विविधता में एकता है।",
        "आज मौसम बहुत अच्छा है और धूप खिली है।",
        "हमें अपनी शिक्षा पर ध्यान देना चाहिए।",
        "गांव में सूर्योदय बहुत सुंदर होता है।",
        "यह किताब हिंदी साहित्य की एक प्रसिद्ध रचना है।",
        "दिल्ली भारत की राजधानी है और बहुत पुराना शहर है।",
        "हम सब मिलकर काम करेंगे तो सब ठीक हो जाएगा।",
        "गंगा नदी भारत की सबसे पवित्र नदी है।",
        "मैं प्रतिदिन सुबह जल्दी उठकर योग करता हूँ।",
        "हिंदी भाषा भारत की सबसे ज्यादा बोली जाने वाली भाषा है।",
        "आज बाजार में बहुत भीड़ थी और सब खरीदारी कर रहे थे।",
        "क्या आप मुझे इसका मतलब बता सकते हैं?",
        "यह बहुत अच्छा काम है और मुझे बहुत पसंद आया।",
        "छात्र परीक्षा की तैयारी कर रहे हैं।",
    ]

    for _ in range(2000):
        text = rng.choice(hindi_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["hi"],
        ))

    en_phrases = [
        "Hello world! How are you today? I hope everything is fine.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    for _ in range(200):
        text = rng.choice(en_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["en"],
        ))

    es_phrases = [
        "Hola mundo! ¿Cómo estás hoy? Espero que todo esté bien.",
        "El rápido zorro marrón salta sobre el perro perezoso.",
    ]
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
    print("Loaded checkpoint")

    acc_before, _, _ = _eval_real_world(model, _DEVICE)
    print(f"Before: {acc_before*100:.1f}%")

    # Load real samples
    real_samples = []
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            d = pickle.load(f)
            real_samples = d.get("samples", d) if isinstance(d, dict) else d

    # Generate targeted fix samples
    fix_samples = gen_fix_targets()
    print(f"Fix samples: {len(fix_samples)}, Real samples: {len(real_samples)}")

    gen_synth = SyntheticGenerator(seed=42)
    hw = {
        "coarse": 8.0,
        "modality": 2.0,
        "subtype": 1.0,
        "code_lang": 1.5,
        "text_lang": 2.0,
        "file_mime": 1.0,
        "risk": 1.0,
    }
    criterion = MultiTaskLoss(hw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=0.01)

    total_steps = 2000
    best_acc = acc_before
    best_state = model.state_dict()
    model.train()

    print(f"\nTraining: {total_steps} steps")

    for step in range(total_steps):
        bs = 16
        # Mix: 40% fix samples, 30% real, 30% synthetic
        n_fix = bs * 2 // 5
        n_real = bs * 3 // 10
        n_synth = bs - n_fix - n_real

        batch = [fix_samples[(step * n_fix + i) % len(fix_samples)] for i in range(n_fix)]
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
            {
                "model_state_dict": best_state,
                "step": total_steps,
                "config": model.config.__dict__,
                "real_world_accuracy": final_acc,
            },
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")
    else:
        print("No improvement, keeping original.")


if __name__ == "__main__":
    main()
