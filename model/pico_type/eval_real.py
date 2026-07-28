"""Evaluate pico-type on real data: The Heap (code) + Wikipedia (text)."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.pico_type.arch import PicoType
from model.pico_type.labels import CODE_LANG_LABELS, TEXT_LANG_LABELS, decode_output
from model.pico_type.data import MAX_BYTES


DATA_DIR = Path(__file__).parent / "data" / "real"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_heap_eval(num_per_lang: int = 50, seed: int = 42) -> list[dict]:
    """Collect contamination-free code eval samples from The Heap."""
    import random
    from datasets import load_dataset

    rng = random.Random(seed)
    # Map Heap language names to our labels
    lang_map = {
        "Python": "python", "JavaScript": "javascript", "TypeScript": "typescript",
        "Java": "java", "Kotlin": "kotlin", "Scala": "scala",
        "C": "c", "C++": "cpp", "C#": "csharp",
        "Go": "go", "Rust": "rust",
        "Ruby": "ruby", "PHP": "php", "Perl": "perl",
        "Swift": "swift", "Dart": "dart",
        "Haskell": "haskell", "OCaml": "ocaml",
        "Erlang": "erlang", "Elixir": "elixir",
        "Clojure": "clojure",
        "Shell": "bash", "PowerShell": "powershell",
        "R": "r", "Julia": "julia",
        "Lua": "lua",
        "SQL": "sql",
        "HTML": "html", "CSS": "css",
    }
    supported = set(CODE_LANG_LABELS)
    samples, counts = [], {}
    for heap_lang in lang_map:
        our_lang = lang_map[heap_lang]
        if our_lang not in supported:
            continue
        try:
            ds = load_dataset("AISE-TUDelft/the-heap", heap_lang, split="train", streaming=True)
            count = 0
            for ex in ds:
                content = ex.get("content", "")
                if not content or len(content.strip()) < 30:
                    continue
                data = content.encode("utf-8")[:MAX_BYTES]
                samples.append({"data": data, "language": our_lang})
                counts[our_lang] = counts.get(our_lang, 0) + 1
                count += 1
                if count >= num_per_lang:
                    break
        except Exception as e:
            print(f"  {heap_lang}: {e}")
    rng.shuffle(samples)
    print(f"Collected {len(samples)} Heap eval samples across {len(counts)} langs")
    return samples


def collect_wikipedia_eval(num_per_lang: int = 50) -> list[dict]:
    """Collect text eval samples from Wikipedia (held-out)."""
    from datasets import load_dataset

    TEXT_LANGS = [
        "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da",
        "fi", "pl", "cs", "sk", "hu", "ro", "el", "tr",
        "ru", "uk", "bg", "sr", "hr",
        "zh", "ja", "ko", "vi", "th", "id", "ms",
    ]
    supported = set(TEXT_LANG_LABELS)
    samples, counts = [], {}
    for iso_lang in TEXT_LANGS:
        if iso_lang not in supported:
            continue
        config = f"20231101.{iso_lang}"
        try:
            ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
            count = 0
            for ex in ds:
                text = ex.get("text", "")
                if not text or len(text.strip()) < 50:
                    continue
                data = text.encode("utf-8")[:MAX_BYTES]
                samples.append({"data": data, "language": iso_lang})
                counts[iso_lang] = counts.get(iso_lang, 0) + 1
                count += 1
                if count >= num_per_lang:
                    break
        except Exception as e:
            print(f"  {iso_lang}: {e}")
    print(f"Collected {len(samples)} text eval samples across {len(counts)} langs")
    return samples


def save_eval_set(samples: list[dict], name: str):
    path = DATA_DIR / f"{name}_eval.json"
    with open(path, "w") as f:
        json.dump([(s["data"].hex(), s["language"]) for s in samples], f)
    print(f"Saved {len(samples)} {name} eval samples to {path}")


def load_eval_set(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}_eval.json"
    if not path.exists():
        return []
    with open(path) as f:
        return [{"data": bytes.fromhex(h), "language": lang} for h, lang in json.load(f)]


def accuracy(predictions: list[str], labels: list[str]) -> float:
    correct = sum(1 for p, lbl in zip(predictions, labels) if p == lbl)
    return correct / max(len(labels), 1) * 100


@torch.no_grad()
def evaluate(model: PicoType, samples: list[dict], device: torch.device, head: str, label_map: dict) -> dict:
    model.eval()
    preds, truths = [], []
    lang_counts = Counter()

    for s in samples:
        data = s["data"]
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        if len(data) < 10:
            continue
        ids = torch.tensor([list(data)], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool, device=device)
        logits = model(ids, mask, tier="base")
        decoded = decode_output(logits, tier="base", undetected_threshold=0.0)
        predicted = decoded.get(head, "undetected")
        target = s["language"]
        preds.append(predicted)
        truths.append(target)
        lang_counts[target] = lang_counts.get(target, 0) + 1

    overall = accuracy(preds, truths)
    per_lang = {}
    for lang in set(truths):
        mask = [t == lang for t in truths]
        lang_preds = [p for p, t in zip(preds, truths) if t == lang]
        lang_truths = [t for t in truths if t == lang]
        if lang_preds:
            per_lang[lang] = accuracy(lang_preds, lang_truths)

    return {
        "overall_accuracy": overall,
        "total_samples": len(truths),
        "languages": len(lang_counts),
        "per_language_accuracy": dict(sorted(per_lang.items(), key=lambda x: -x[1])),
    }


def main(checkpoint_dir: str = "checkpoints"):
    print(f"Loading model from {checkpoint_dir}...")
    from model.pico_type.arch import PicoTypeConfig

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Find latest checkpoint
    ckpts = sorted(Path(checkpoint_dir).glob("*.pt"), key=os.path.getmtime)
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    ckpt_path = ckpts[-1]
    print(f"Using checkpoint: {ckpt_path}")

    model = PicoType(PicoTypeConfig()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded step {ckpt.get('step', '?')}, eval_loss={ckpt.get('eval_loss', 'N/A')}")

    # Load or collect eval sets
    heap = load_eval_set("code")
    if not heap:
        print("\n=== Collecting Heap eval samples ===")
        heap = collect_heap_eval(50)
        save_eval_set(heap, "code")

    wikipedia = load_eval_set("text")
    if not wikipedia:
        print("\n=== Collecting Wikipedia eval samples ===")
        wikipedia = collect_wikipedia_eval(50)
        save_eval_set(wikipedia, "text")

    # Evaluate
    print("\n=== Code Language Detection (The Heap) ===")
    code_results = evaluate(model, heap, device, "code_language", {})
    print(f"Overall accuracy: {code_results['overall_accuracy']:.1f}% ({code_results['total_samples']} samples, {code_results['languages']} langs)")
    worst = list(code_results["per_language_accuracy"].items())[:5]
    print(f"Worst languages: {worst}")

    print("\n=== Text Language Detection (Wikipedia) ===")
    text_results = evaluate(model, wikipedia, device, "text_language", {})
    print(f"Overall accuracy: {text_results['overall_accuracy']:.1f}% ({text_results['total_samples']} samples, {text_results['languages']} langs)")
    worst = list(text_results["per_language_accuracy"].items())[:5]
    print(f"Worst languages: {worst}")

    # Results summary
    results = {
        "checkpoint": str(ckpt_path),
        "step": ckpt.get("step", 0),
        "code": code_results,
        "text": text_results,
    }
    results_path = DATA_DIR / f"eval_results_{checkpoint_dir.replace('/', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    main(checkpoint_dir)
