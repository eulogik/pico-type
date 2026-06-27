"""Round 8: Long multi-head training to surpass 95% RWA.

Strategy:
- All heads active (prevents catastrophic forgetting)
- 50% real data, 30% targeted synthetic, 20% general synthetic
- Targeted: Hindi, French, YAML config, traceback patterns
- General: diverse code, text, markup, config, error patterns
- Save best checkpoint every eval
- 10000 steps (~3.5h on CPU)
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


def gen_targeted_samples(rng=random.Random(42)):
    """Generate diverse samples targeting all failure patterns."""
    samples = []

    # --- Hindi text (was failing) ---
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
        "पानी पृथ्वी पर सबसे महत्वपूर्ण संसाधन है।",
        "हरे पेड़ पर्यावरण के लिए बहुत जरूरी हैं।",
        "बच्चे खेल में बहुत मज़े कर रहे हैं।",
        "यह शहर बहुत सुंदर और स्वच्छ है।",
        "कृपया मेरी मदद करें यह बहुत जरूरी है।",
    ]
    for _ in range(500):
        text = rng.choice(hindi_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["hi"],
        ))

    # --- French text (currently detected as Spanish) ---
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
        "Le president de la Republique a fait une declaration importante.",
        "Les eleves preparent leurs examens avec beaucoup de serieux.",
        "Le printemps est la saison la plus agreeable de l'annee.",
        "Je lis un livre tres interessant sur l'histoire de France.",
        "La cuisine francaise est reconnue dans le monde entier.",
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
    for _ in range(500):
        text = rng.choice(fr_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))
    for _ in range(100):
        text = rng.choice(es_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["es"],
        ))

    # --- YAML config (was misclassified as error) ---
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
        "name: {k}\nversion: {v}\ndevDependencies:\n  pytest: ^7.0\n  ruff: ^0.1.0\n",
        "docker:\n  image: {k2}:{v}\n  ports:\n    - {v2}:8080\n",
        "terraform:\n  required_version: {v}\n  providers:\n    aws:\n      region: us-east-1\n",
    ]
    for _ in range(500):
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

    # --- Traceback patterns (expected "trace" but label is "error") ---
    for _ in range(300):
        fn = _id(rng, 5) + ".py"
        func = _id(rng, 6)
        exc = rng.choice(["ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError"])
        msg = _id(rng, 10)
        v = f"Traceback (most recent call last):\n  File \"{fn}\", line 3, in <module>\n    {func}()\n  File \"{fn}\", line 5, in {func}\n    return {msg}\n{exc}: {msg}\n"
        samples.append(Sample(
            data=v.encode("utf-8"),
            coarse=_COARSE["error"],
            modality=_MODALITY["textual"],
        ))

    # --- English text distractors ---
    en_phrases = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Machine learning models require large datasets for training.",
        "The weather forecast predicts rain for the next three days.",
        "Scientists have discovered a new species of butterfly in the Amazon.",
        "The stock market experienced significant volatility this week.",
    ]
    for _ in range(300):
        text = rng.choice(en_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["en"],
        ))

    # --- Code samples for code_lang balance ---
    code_templates = {
        "python": [
            "def {f}({a}):\n    return {a} * 2\n\ndef main():\n    print({f}(5))\n",
            "import os\nimport sys\n\ndef {f}():\n    path = os.path.join('.', '{a}')\n    return os.path.exists(path)\n",
            "class {f}:\n    def __init__(self, {a}):\n        self.{a} = {a}\n    def get(self):\n        return self.{a}\n",
        ],
        "javascript": [
            "function {f}({a}) {{\n    return {a} * 2;\n}}\n\nconsole.log({f}(5));\n",
            "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {{\n    res.json({{ message: '{f}' }});\n}});\n",
            "const {a} = {{ name: '{f}', value: 42 }};\nconsole.log({a}.name);\n",
        ],
        "java": [
            "public class {f} {{\n    public static int {a}(int x) {{\n        return x * 2;\n    }}\n    public static void main(String[] args) {{\n        System.out.println({a}(5));\n    }}\n}}\n",
        ],
        "rust": [
            "fn {f}({a}: i32) -> i32 {{\n    {a} * 2\n}}\n\nfn main() {{\n    println!(\"{{}}\", {f}(5));\n}}\n",
        ],
        "go": [
            "package main\n\nimport \"fmt\"\n\nfunc {f}({a} int) int {{\n    return {a} * 2\n}}\n\nfunc main() {{\n    fmt.Println({f}(5))\n}}\n",
        ],
        "bash": [
            "#!/bin/bash\nfor {a} in *.txt; do\n    echo \"Processing ${{{a}}}\"\n    wc -l \"${{{a}}}\"\ndone\n",
        ],
        "sql": [
            "SELECT u.id, u.name, u.email\nFROM users u\nWHERE u.active = 1\nORDER BY u.name\nLIMIT 100;\n",
        ],
        "typescript": [
            "interface {f} {{\n    name: string;\n    value: number;\n}}\n\nconst {a}: {f} = {{ name: '{f}', value: 42 }};\nconsole.log({a}.name);\n",
        ],
        "kotlin": [
            "fun {f}({a}: Int): Int {{\n    return {a} * 2\n}}\n\nfun main() {{\n    println({f}(5))\n}}\n",
        ],
        "php": [
            "<?php\nfunction {f}(${a}) {{\n    return ${a} * 2;\n}}\n\necho {f}(5);\n?>\n",
        ],
        "ruby": [
            "def {f}({a})\n    {a} * 2\nend\n\nputs {f}(5)\n",
        ],
    }
    for lang, templates in code_templates.items():
        if lang not in _CODE:
            continue
        for _ in range(50):
            tmpl = rng.choice(templates)
            text = tmpl.format(f=_id(rng, 5), a=_id(rng, 3))
            samples.append(Sample(
                data=text.encode("utf-8")[:2048],
                coarse=_COARSE["code"],
                modality=_MODALITY["textual"],
                code_lang=_CODE[lang],
            ))

    # --- JSON config ---
    for _ in range(200):
        keys = [_id(rng, 5) for _ in range(3)]
        vals = [str(rng.randint(1, 999)), _id(rng, 4), rng.choice(["true", "false"])]
        pairs = ", ".join(f'"{k}": "{v}"' for k, v in zip(keys, vals))
        text = "{ " + pairs + " }\n"
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["config"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE["json"],
        ))

    # --- XML/HTML markup ---
    for _ in range(200):
        tag = rng.choice(["div", "span", "p", "h1", "h2", "ul", "li"])
        content = _id(rng, 10)
        text = f"<{tag} class=\"container\">\n  <{tag}>{content}</{tag}>\n</{tag}>\n"
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["markup"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE.get("html", 0),
        ))

    rng.shuffle(samples)
    return samples


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")
    backup_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best_90.5.pt")

    # Backup current best
    import shutil
    shutil.copy2(ckpt_path, backup_path)
    print(f"Backed up current best.pt to {backup_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    acc_before, _, _ = _eval_real_world(model, _DEVICE)
    print(f"Starting RWA: {acc_before*100:.1f}%")

    # Load real samples
    real_samples = []
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            d = pickle.load(f)
            real_samples = d.get("samples", d) if isinstance(d, dict) else d
    print(f"Real samples: {len(real_samples)}")

    # Generate targeted samples
    targeted = gen_targeted_samples()
    print(f"Targeted samples: {len(targeted)}")

    gen_synth = SyntheticGenerator(seed=42)

    hw = {
        "coarse": 8.0,
        "modality": 2.0,
        "subtype": 2.0,
        "code_lang": 2.0,
        "text_lang": 3.0,
        "file_mime": 1.0,
        "risk": 1.0,
    }
    criterion = MultiTaskLoss(hw)

    total_steps = 10000
    best_acc = acc_before
    best_state = model.state_dict()
    best_step = 0

    # Learning rate schedule: warmup 200 steps, then cosine decay
    def get_lr(step):
        if step < 200:
            return 3e-6 * (step + 1) / 200
        progress = (step - 200) / (total_steps - 200)
        return 3e-6 * 0.3 * (1 + math.cos(progress * math.pi))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=0.01)
    model.train()

    print(f"\nStarting {total_steps} steps on {_DEVICE}")
    print("Mix: 40% targeted, 30% real, 30% synthetic")
    start_time = time.time()

    for step in range(total_steps):
        bs = 16
        n_targeted = bs * 2 // 5  # 40%
        n_real = bs * 3 // 10     # 30%
        n_synth = bs - n_targeted - n_real  # 30%

        batch = [targeted[(step * n_targeted + i) % len(targeted)] for i in range(n_targeted)]
        if n_real > 0 and real_samples:
            ri = (step * n_real) % len(real_samples)
            batch.extend(real_samples[ri:ri + n_real])
        batch.extend([gen_synth() for _ in range(n_synth)])

        coll = collate_fn(batch)
        ids = coll["input_ids"].to(_DEVICE)
        mask = coll["attention_mask"].to(_DEVICE)
        labels = {k: v.to(_DEVICE) for k, v in coll["labels"].items()}

        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, ("base",))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (total_steps - step - 1) / rate if rate > 0 else 0
            parts = [f"step={step:5d}/{total_steps} lr={lr:.7f} loss={loss.item():.4f} [{elapsed:.0f}s, ETA {eta:.0f}s]"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        if step % 500 == 0 and step > 0:
            acc, cor, tot = _eval_real_world(model, _DEVICE)
            elapsed = time.time() - start_time
            print(f"  EVAL step={step}: RWA={acc*100:.1f}% ({cor}/{tot}) [{elapsed:.0f}s elapsed]")
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                best_step = step
                print(f"  *** NEW BEST: {acc*100:.1f}% ***")
                # Save intermediate best
                torch.save({
                    "model_state_dict": best_state,
                    "step": step,
                    "config": model.config.__dict__,
                    "real_world_accuracy": best_acc,
                }, ckpt_path)
            model.train()

    # Final eval
    model.load_state_dict(best_state)
    model.eval()
    final_acc, cor, tot = _eval_real_world(model, _DEVICE)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best RWA: {best_acc*100:.1f}% ({best_step} steps)")
    print(f"Final RWA: {final_acc*100:.1f}% ({cor}/{tot})")
    print(f"{'='*60}")

    if final_acc >= acc_before:
        torch.save({
            "model_state_dict": best_state,
            "step": best_step,
            "config": model.config.__dict__,
            "real_world_accuracy": final_acc,
        }, ckpt_path)
        print(f"Saved best.pt: {final_acc*100:.1f}%")
    else:
        print("Restoring backup (no improvement)")
        shutil.copy2(backup_path, ckpt_path)


if __name__ == "__main__":
    main()
