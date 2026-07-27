"""Collect real text samples for training/distillation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from model.pico_type.data import TEXT_LANG_LABELS, LABEL_MAP_TEXT

CACHE_DIR = Path(__file__).parent / "data" / "real"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Map ISO codes to OSCAR/CommonCrawl language codes
ISO_TO_DATASET_LANG = {
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "ru": "ru",
    "zh": "zh", "ja": "ja", "ko": "ko",
    "ar": "ar", "hi": "hi", "bn": "bn",
    "ur": "ur", "vi": "vi", "th": "th",
    "id": "id", "ms": "ms", "tl": "tl",
    "sw": "sw",
}

SUPPORTED_LANGS = set(TEXT_LANG_LABELS) & set(ISO_TO_DATASET_LANG.keys())


def collect_from_oscar(
    num_samples: int = 5000,
    max_length: int = 1024,
    seed: int = 42,
) -> List[Dict]:
    """Collect text samples from OSCAR/CommonCrawl."""
    rng = random.Random(seed)
    samples: List[Dict] = []
    per_lang = max(1, num_samples // len(SUPPORTED_LANGS))

    try:
        from datasets import load_dataset

        lang_counts: Dict[str, int] = {}
        for iso_lang in SUPPORTED_LANGS:
            ds_lang = ISO_TO_DATASET_LANG[iso_lang]
            try:
                ds = load_dataset(
                    "oscar-corpus/OSCAR-2301",
                    language=ds_lang,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                needed = per_lang - lang_counts.get(iso_lang, 0)
                count = 0
                for example in ds:
                    text = example.get("text", "")
                    if not text or len(text.strip()) < 20:
                        continue
                    if len(text.encode("utf-8")) > max_length:
                        text = text[:max_length]
                    samples.append({"data": text.encode("utf-8"), "language": iso_lang})
                    lang_counts[iso_lang] = lang_counts.get(iso_lang, 0) + 1
                    count += 1
                    if count >= needed:
                        break
            except Exception as e:
                print(f"  {iso_lang}: error - {e}")
            if len(samples) >= num_samples:
                break
    except ImportError:
        print("datasets not installed. Install with: pip install datasets")
    except Exception as e:
        print(f"Error: {e}")

    rng.shuffle(samples)
    return samples[:num_samples]


def save_samples(samples: List[Dict], name: str = "text"):
    path = CACHE_DIR / f"{name}_samples.json"
    serializable = [(s["data"].hex(), s["language"]) for s in samples]
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"Saved {len(samples)} {name} samples to {path}")
    return path


def load_samples(name: str = "text") -> List[Dict]:
    path = CACHE_DIR / f"{name}_samples.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [{"data": bytes.fromhex(h), "lang": lang} for h, lang in data]


def collect_and_save(num_samples: int = 5000):
    print(f"Collecting {num_samples} real text samples...")
    samples = collect_from_oscar(num_samples=num_samples)
    if samples:
        save_samples(samples)
    print(f"Got {len(samples)} samples across {len(set(s['language'] for s in samples))} languages")
    return samples


if __name__ == "__main__":
    collect_and_save(2500)
