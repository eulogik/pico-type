"""Collect real training data from HuggingFace datasets."""

from __future__ import annotations

import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data" / "real"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main_code(num_samples: int = 12000):
    from model.pico_type.collect_code import run as cc
    cc(num_samples=num_samples)


def main_text(num_samples: int = 10000):
    from model.pico_type.collect_text import run as ct
    ct(num_samples=num_samples)


def main_all(num_code: int = 12000, num_text: int = 10000):
    print("=" * 60)
    print("Collecting real training data for pico-type v2")
    print("=" * 60)
    main_code(num_code)
    main_text(num_text)
    print("\nDone collecting real data.")
    print(f"  Data dir: {DATA_DIR}")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    n_code = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    n_text = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"

    if mode == "code":
        main_code(n_code)
    elif mode == "text":
        main_text(n_text)
    else:
        main_all(n_code, n_text)
