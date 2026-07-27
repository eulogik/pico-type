"""Collect real code samples from public HF datasets."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List

CACHE_DIR = Path(__file__).parent / "data" / "real"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXT_MAP = {
    ".py": "python", ".pyw": "python", ".pyx": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".sc": "scala",
    ".groovy": "groovy", ".gvy": "groovy",
    ".clj": "clojure", ".cljs": "clojure",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".fs": "fsharp", ".fsx": "fsharp",
    ".m": "matlab", ".mm": "objectivec",
    ".go": "go",
    ".rs": "rust",
    ".zig": "zig",
    ".rb": "ruby",
    ".php": "php",
    ".pl": "perl", ".pm": "perl",
    ".lua": "lua",
    ".tcl": "tcl",
    ".swift": "swift",
    ".dart": "dart",
    ".jl": "julia",
    ".nim": "nim",
    ".cr": "crystal",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".elm": "elm",
    ".erl": "erlang",
    ".ex": "elixir",
    ".lisp": "lisp", ".cl": "lisp",
    ".scm": "scheme",
    ".rkt": "racket",
    ".r": "r", ".R": "r",
    ".sas": "sas",
    ".sql": "sql",
    ".html": "html", ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".sh": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".ps1": "powershell",
    ".vim": "vim",
    ".f90": "fortran", ".f": "fortran",
    ".cbl": "cobol",
    ".ada": "ada", ".adb": "ada",
    ".pas": "pascal",
    ".vb": "vb",
    ".pro": "prolog",
    ".vhd": "vhdl",
    ".pkb": "plsql",
}
SUPPORTED = sorted(set(EXT_MAP.values()))


def infer_lang(file_path: str) -> str | None:
    _, ext = os.path.splitext(file_path.lower())
    return EXT_MAP.get(ext) if ext else None


def collect_code(
    num_samples: int = 12000, max_length: int = 1024, seed: int = 42
) -> List[Dict]:
    rng = random.Random(seed)
    base = max(1, num_samples // len(SUPPORTED))
    targets = {lang: base for lang in SUPPORTED}
    for p in ["python", "javascript", "typescript", "java", "cpp", "go", "rust", "ruby", "php", "c"]:
        if p in targets:
            targets[p] = int(targets[p] * 2)

    samples, counts = [], {}

    from datasets import load_dataset
    ds = load_dataset("nick007x/github-code-2025", split="train", streaming=True)
    ds = ds.take(200000)
    print("Taking 200K samples from github-code-2025...")

    for i, ex in enumerate(ds):
        lang = infer_lang(ex.get("file_path", ""))
        if lang is None:
            continue
        if counts.get(lang, 0) >= targets[lang]:
            continue
        content = ex.get("content", "")
        if not content or len(content.strip()) < 30:
            continue
        data = content.encode("utf-8")[:max_length]
        samples.append({"data": data, "language": lang})
        counts[lang] = counts.get(lang, 0) + 1
        if len(samples) >= num_samples:
            break

    rng.shuffle(samples)
    print(f"Got {len(samples)} samples across {len(counts)} langs")
    for lang, c in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {lang:15s}: {c}")
    missing = set(SUPPORTED) - set(counts)
    if missing:
        print(f"Missing ({len(missing)}): {sorted(missing)}")
    return samples


def save(samples, name="code"):
    path = CACHE_DIR / f"{name}_samples.json"
    with open(path, "w") as f:
        json.dump([(s["data"].hex(), s["language"]) for s in samples], f)
    print(f"Saved {len(samples)} samples to {path}")


def load(name="code"):
    path = CACHE_DIR / f"{name}_samples.json"
    if not path.exists():
        return []
    with open(path) as f:
        return [{"data": bytes.fromhex(h), "language": lang} for h, lang in json.load(f)]


def run(num_samples=12000):
    print(f"\n=== Collecting {num_samples} code samples ===")
    samples = collect_code(num_samples=num_samples)
    if samples:
        save(samples)
    return samples


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
