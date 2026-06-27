"""Round 9: Ultra-conservative training for the 2 fixable failures.

Failures:
  #4: Java code → "link" (coarse error)
  #14: JSON config → "code" (coarse error)

Strategy:
- CPU only (stable gradients, no MPS noise)
- Ultra-low LR: 1e-6 with gentle cosine decay
- 70% real data, 15% targeted (Java + JSON), 15% general synthetic
- All heads active to prevent forgetting
- Save checkpoints every 500 steps, keep the best
"""

from __future__ import annotations

import math
import os
import pickle
import random
import shutil
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

_DEVICE = torch.device("cpu")  # CPU for stability


def _id(rng, n=8):
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n))


def gen_targeted(rng=random.Random(77)):
    """Targeted samples for the 2 failures + nearby patterns."""
    samples = []

    # --- Java code (should be "code" not "link") ---
    java_templates = [
        "public class {cls} {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello\");\n    }}\n}}\n",
        "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {cls} {{\n    public static void main(String[] args) {{\n        List<String> items = new ArrayList<>();\n        items.add(\"test\");\n        System.out.println(items.size());\n    }}\n}}\n",
        "public class {cls} {{\n    private String name;\n    \n    public {cls}(String name) {{\n        this.name = name;\n    }}\n    \n    public String getName() {{\n        return name;\n    }}\n}}\n",
        "public class {cls} {{\n    public int add(int a, int b) {{\n        return a + b;\n    }}\n}}\n",
        "import java.util.stream.IntStream;\n\npublic class {cls} {{\n    public static void main(String[] args) {{\n        int sum = IntStream.range(1, 101).sum();\n        System.out.println(sum);\n    }}\n}}\n",
        "public class {cls} {{\n    private static final int MAX_SIZE = 100;\n    \n    public static void main(String[] args) {{\n        int[] arr = new int[MAX_SIZE];\n        for (int i = 0; i < MAX_SIZE; i++) {{\n            arr[i] = i * 2;\n        }}\n    }}\n}}\n",
    ]
    for _ in range(200):
        tmpl = rng.choice(java_templates)
        text = tmpl.format(cls=_id(rng, 5).capitalize())
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["code"],
            modality=_MODALITY["textual"],
            code_lang=_CODE["java"],
        ))

    # --- Also nearby: Kotlin, Scala (similar to Java) ---
    kotlin_templates = [
        "fun main() {{\n    val items = listOf(\"a\", \"b\", \"c\")\n    println(items.size)\n}}\n",
        "data class {cls}(val name: String, val value: Int)\n\nfun main() {{\n    val item = {cls}(\"test\", 42)\n    println(item.name)\n}}\n",
    ]
    for _ in range(50):
        tmpl = rng.choice(kotlin_templates)
        text = tmpl.format(cls=_id(rng, 5).capitalize())
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["code"],
            modality=_MODALITY["textual"],
            code_lang=_CODE.get("kotlin", 0),
        ))

    # --- JSON config (should be "config" not "code") ---
    for _ in range(200):
        keys = [_id(rng, 5) for _ in range(rng.randint(3, 6))]
        vals = [str(rng.randint(1, 999)), _id(rng, 4), rng.choice(["true", "false"]), str(rng.randint(10, 9999))]
        pairs = []
        for k, v in zip(keys, vals):
            pairs.append(f'    "{k}": "{v}"')
        text = "{\n" + ",\n".join(pairs) + "\n}\n"
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["config"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE["json"],
        ))

    # --- YAML config (prevent regression) ---
    yaml_bases = [
        "name: {k}\nversion: {v}\ndescription: {k2}\n",
        "{k}: {v}\n{k2}: {v2}\n",
        "app:\n  name: {k}\n  version: {v}\n  env: {v2}\n",
    ]
    for _ in range(100):
        tmpl = rng.choice(yaml_bases)
        text = tmpl.format(k=_id(rng, 5), v=str(rng.randint(1, 999)), k2=_id(rng, 6), v2=rng.choice(["dev", "prod", "test"]))
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["config"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE["yaml"],
        ))

    # --- Hindi text (prevent regression) ---
    hindi_phrases = [
        "नमस्ते दुनिया! आप कैसे हैं?",
        "भारत एक महान देश है।",
        "आज मौसम बहुत अच्छा है।",
        "हमें अपनी शिक्षा पर ध्यान देना चाहिए।",
    ]
    for _ in range(100):
        text = rng.choice(hindi_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["hi"],
        ))

    # --- French text (prevent regression) ---
    fr_phrases = [
        "Bonjour le monde! Comment allez-vous?",
        "Je m'appelle Pierre et j'habite a Paris.",
        "La vie est belle quand on sait apprecier.",
        "Le developpement web est passionnant.",
    ]
    for _ in range(100):
        text = rng.choice(fr_phrases)
        samples.append(Sample(
            data=text.encode("utf-8"),
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=_TEXT["fr"],
        ))

    rng.shuffle(samples)
    return samples


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best.pt")
    real_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "real_samples.pkl")
    backup_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best_90.5_round9.pt")
    best_snapshot = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best_snapshot.pt")

    # Backup current best
    shutil.copy2(ckpt_path, backup_path)
    print(f"Backed up to {backup_path}")

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

    targeted = gen_targeted()
    print(f"Targeted samples: {len(targeted)}")

    gen_synth = SyntheticGenerator(seed=42)

    hw = {
        "coarse": 10.0,  # Higher coarse weight to prevent forgetting
        "modality": 2.0,
        "subtype": 2.0,
        "code_lang": 1.5,
        "text_lang": 1.5,
        "file_mime": 1.0,
        "risk": 1.0,
    }
    criterion = MultiTaskLoss(hw)

    total_steps = 5000
    best_acc = acc_before
    best_state = model.state_dict()
    best_step = 0
    eval_count = 0

    def get_lr(step):
        if step < 200:
            return 1e-6 * (step + 1) / 200
        progress = (step - 200) / (total_steps - 200)
        return 1e-6 * 0.3 * (1 + math.cos(progress * math.pi))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    model.train()

    print(f"\nStarting {total_steps} steps on CPU (ultra-conservative)")
    print("Mix: 70% real, 15% targeted, 15% synthetic")
    print("LR: 1e-6 max, cosine decay")
    start_time = time.time()

    for step in range(total_steps):
        bs = 16
        n_real = int(bs * 0.70)   # 70% real
        n_targeted = int(bs * 0.15)  # 15% targeted
        n_synth = bs - n_real - n_targeted  # 15% synthetic

        batch = [real_samples[(step * n_real + i) % len(real_samples)] for i in range(n_real)]
        batch += [targeted[(step * n_targeted + i) % len(targeted)] for i in range(n_targeted)]
        batch += [gen_synth() for _ in range(n_synth)]

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

        if step % 500 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (total_steps - step - 1) / rate if rate > 0 else 0
            parts = [f"step={step:5d}/{total_steps} lr={lr:.8f} loss={loss.item():.4f} [{elapsed:.0f}s ETA {eta:.0f}s]"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        if step % 500 == 0 and step > 0:
            acc, cor, tot = _eval_real_world(model, _DEVICE)
            elapsed = time.time() - start_time
            eval_count += 1
            print(f"  EVAL #{eval_count} step={step}: RWA={acc*100:.1f}% ({cor}/{tot}) [{elapsed:.0f}s]")
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                best_step = step
                print(f"  *** NEW BEST: {acc*100:.1f}% ***")
                torch.save({
                    "model_state_dict": best_state,
                    "step": step,
                    "config": model.config.__dict__,
                    "real_world_accuracy": best_acc,
                }, ckpt_path)
                # Also save a snapshot
                shutil.copy2(ckpt_path, best_snapshot)
            model.train()

    # Final eval
    model.load_state_dict(best_state)
    model.eval()
    final_acc, cor, tot = _eval_real_world(model, _DEVICE)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best RWA: {best_acc*100:.1f}% (step {best_step})")
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
        print("No improvement, restoring backup")
        shutil.copy2(backup_path, ckpt_path)


if __name__ == "__main__":
    main()
