"""Collect real text samples from Wikipedia (323 langs, clean, public)."""

from __future__ import annotations

import json
import random
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "real"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEXT_LANGS = [
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da",
    "fi", "pl", "cs", "sk", "hu", "ro", "el", "tr",
    "ru", "uk", "bg", "sr", "hr",
    "zh", "ja", "ko", "vi", "th", "id", "ms",
]
WIKIPEDIA_PREFIX = "20231101"


def collect_text(
    num_samples: int = 10000,
    max_length: int = 1024,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    counts: dict[str, int] = {}
    base = max(1, num_samples // len(TEXT_LANGS))
    targets = {lang: base for lang in TEXT_LANGS}
    for popular in ["en", "es", "fr", "de", "zh", "ja", "ru"]:
        if popular in targets:
            targets[popular] = int(targets[popular] * 1.5)

    try:
        from datasets import load_dataset

        for iso_lang in TEXT_LANGS:
            needed = targets[iso_lang]
            if needed <= 0:
                continue
            config = f"{WIKIPEDIA_PREFIX}.{iso_lang}"
            try:
                ds = load_dataset(
                    "wikimedia/wikipedia", config, split="train", streaming=True
                )
            except Exception as e:
                # Try alternate codes
                alt = {"no": "nb", "hr": "sh", "sr": "sh"}.get(iso_lang)
                if alt:
                    config = f"{WIKIPEDIA_PREFIX}.{alt}"
                    try:
                        ds = load_dataset(
                            "wikimedia/wikipedia", config, split="train", streaming=True
                        )
                    except Exception as e2:
                        print(f"  {iso_lang}: {e2}")
                        continue
                else:
                    print(f"  {iso_lang}: {e}")
                    continue

            count = 0
            for example in ds:
                text = example.get("text", "")
                if not text or len(text.strip()) < 50:
                    continue
                data = text.encode("utf-8")[:max_length]
                samples.append({"data": data, "language": iso_lang})
                counts[iso_lang] = counts.get(iso_lang, 0) + 1
                count += 1
                if count >= needed:
                    break
            print(f"  {iso_lang}: {count} samples")

    except Exception as e:
        print(f"Error: {e}")

    rng.shuffle(samples)
    print(f"\nCollected {len(samples)} samples across {len(counts)}/{len(TEXT_LANGS)} langs")
    for lang, c in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {lang:5s}: {c}")
    missing = set(TEXT_LANGS) - set(counts)
    if missing:
        print(f"Missing: {sorted(missing)}")
    return samples[:num_samples]


def save(samples, name="text"):
    path = CACHE_DIR / f"{name}_samples.json"
    with open(path, "w") as f:
        json.dump([(s["data"].hex(), s["language"]) for s in samples], f)
    print(f"Saved {len(samples)} samples to {path}")


def load(name="text"):
    path = CACHE_DIR / f"{name}_samples.json"
    if not path.exists():
        return []
    with open(path) as f:
        return [{"data": bytes.fromhex(h), "language": lang} for h, lang in json.load(f)]


def run(num_samples=10000):
    print(f"\n=== Collecting {num_samples} text samples from Wikipedia ===")
    samples = collect_text(num_samples=num_samples)
    if samples:
        save(samples)
    return samples


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
