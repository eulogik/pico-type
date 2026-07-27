"""Collect real training data from HuggingFace datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "real"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_code(num_samples: int = 5000):
    from .collect_code import collect_and_save as cc
    cc(num_samples=num_samples)


def collect_text(num_samples: int = 2500):
    from .collect_text import collect_and_save as ct
    ct(num_samples=num_samples)


def collect_all(num_code: int = 5000, num_text: int = 2500):
    print("Collecting real training data...")
    collect_code(num_code)
    collect_text(num_text)
    print("Done collecting real data.")


def get_code_samples() -> list:
    """Load collected code samples with label indices."""
    from .collect_code import load_samples
    from .data import _CODE

    samples = load_samples()
    for s in samples:
        s["label_idx"] = _CODE.get(s["language"], -100)
    return samples


def get_text_samples() -> list:
    """Load collected text samples with label indices."""
    from .collect_text import load_samples
    from .data import _TEXT

    samples = load_samples()
    for s in samples:
        s["label_idx"] = _TEXT.get(s["language"], -100)
    return samples


if __name__ == "__main__":
    import sys
    n_code = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n_text = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    collect_all(n_code, n_text)
