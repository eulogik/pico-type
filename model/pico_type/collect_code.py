"""Collect real code samples for training/distillation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List

from model.pico_type.data import CODE_LANG_LABELS

CACHE_DIR = Path(__file__).parent / "data" / "real"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LANG_ALIASES = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "java": "java",
    "c": "c", "c++": "cpp", "cpp": "cpp", "cplusplus": "cpp",
    "c#": "csharp", "csharp": "csharp",
    "go": "go", "golang": "go",
    "rust": "rust", "rs": "rust",
    "swift": "swift",
    "kotlin": "kotlin", "kt": "kotlin",
    "scala": "scala",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "bash": "bash", "shell": "bash", "sh": "bash", "zsh": "bash",
    "sql": "sql",
    "r": "r", "rlang": "r",
    "julia": "julia", "jl": "julia",
    "perl": "perl", "pl": "perl",
    "haskell": "haskell", "hs": "haskell",
    "lua": "lua",
    "elixir": "elixir", "ex": "elixir",
    "clojure": "clojure", "clj": "clojure",
    "erlang": "erlang", "erl": "erlang",
    "groovy": "groovy",
    "dart": "dart",
    "zig": "zig",
    "nim": "nim",
    "lisp": "lisp", "commonlisp": "lisp",
    "scheme": "scheme", "scm": "scheme",
    "prolog": "prolog",
    "fortran": "fortran", "f90": "fortran",
    "matlab": "matlab",
    "powershell": "powershell", "ps1": "powershell",
    "makefile": "makefile", "make": "makefile", "mk": "makefile",
    "cmake": "cmake",
    "gradle": "gradle",
    "terraform": "terraform", "tf": "terraform",
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml",
    "json": "json",
    "dockerfile": "dockerfile",
    "graphql": "graphql", "gql": "graphql",
    "protobuf": "protobuf", "proto": "protobuf",
    "thrift": "thrift",
    "flatbuffers": "flatbuffers", "fbs": "flatbuffers",
    "ocaml": "ocaml", "ml": "ocaml",
    "objectivec": "objectivec",
    "elm": "elm",
}

SUPPORTED = {v for v in LANG_ALIASES.values() if v in CODE_LANG_LABELS}


def collect_from_hf(
    num_samples: int = 10000,
    max_length: int = 1024,
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    samples: List[Dict] = []
    per_lang = max(1, num_samples // len(SUPPORTED))

    try:
        from datasets import load_dataset

        lang_counts: Dict[str, int] = {}
        ds = load_dataset("codeparrot/github-code", split="train", streaming=True, trust_remote_code=True)
        for example in ds:
            raw = example.get("language", "").lower()
            lang = LANG_ALIASES.get(raw)
            if lang is None or lang not in SUPPORTED:
                continue
            if lang_counts.get(lang, 0) >= per_lang:
                continue
            code = example.get("code", "")
            if not code or len(code.strip()) < 20:
                continue
            data = code.encode("utf-8")[:max_length]
            samples.append({"data": data, "language": lang})
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if len(samples) >= num_samples:
                break
    except ImportError:
        print("Install datasets: pip install datasets")
    except Exception as e:
        print(f"Error: {e}")

    rng.shuffle(samples)
    return samples


def save_samples(samples: List[Dict], name: str = "code"):
    path = CACHE_DIR / f"{name}_samples.json"
    serializable = [(s["data"].hex(), s["language"]) for s in samples]
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"Saved {len(samples)} {name} samples ({len(set(s['language'] for s in samples))} langs) to {path}")
    return path


def load_samples(name: str = "code") -> List[Dict]:
    path = CACHE_DIR / f"{name}_samples.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [{"data": bytes.fromhex(h), "language": lang} for h, lang in data]


def collect_and_save(num_samples: int = 10000):
    print(f"Collecting {num_samples} real code samples...")
    samples = collect_from_hf(num_samples=num_samples)
    if samples:
        save_samples(samples)
    print(f"Got {len(samples)} samples")
    return samples


if __name__ == "__main__":
    collect_and_save(5000)
