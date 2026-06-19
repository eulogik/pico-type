"""Fetch real code samples — fast path: Search API only, skip slow fallback repos."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pico_type.realdata import search_download_code
from model.pico_type.labels import CODE_LANG_LABELS
from model.pico_type.data import MAX_BYTES, MIN_BYTES, Sample

_COARSE = {"code": 2}
_MODALITY = {"textual": 0}


def fetch_fast(
    language: str,
    samples_per_lang: int = 20,
    token: str = "",
) -> list[Sample]:
    """Fast fetch: Search API only, skip slow fallback repos entirely."""
    lang_idx = CODE_LANG_LABELS.index(language)

    # Strategy 1: Search API with high limits (100 per page, up to 200 results)
    downloaded = search_download_code(language, samples_per_lang, token)
    n_search = len(downloaded)
    print(f"  Search API: {n_search}/{samples_per_lang}", end="")

    samples = []
    for content in downloaded:
        samples.append(Sample(
            data=content[:MAX_BYTES],
            coarse=_COARSE["code"],
            modality=_MODALITY["textual"],
            code_lang=lang_idx,
        ))

    if n_search >= samples_per_lang:
        print()
        return samples

    # Strategy 2: Lightweight fallback — only use fast repos
    # Skip expensive tree API calls; just try well-known single files
    known_files = {
        "python": ["https://raw.githubusercontent.com/python/cpython/main/README.rst",
                    "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py"],
        "c": ["https://raw.githubusercontent.com/torvalds/linux/master/README",
              "https://raw.githubusercontent.com/redis/redis/src/server.c"],
        "cpp": ["https://raw.githubusercontent.com/nlohmann/json/single_include/nlohmann/json.hpp",
                "https://raw.githubusercontent.com/ocornut/imgui/imgui.cpp"],
        "javascript": ["https://raw.githubusercontent.com/expressjs/express/lib/express.js"],
        "typescript": ["https://raw.githubusercontent.com/microsoft/TypeScript/src/compiler/core.ts"],
        "java": ["https://raw.githubusercontent.com/spring-projects/spring-boot/main/pom.xml"],
        "rust": ["https://raw.githubusercontent.com/rust-lang/rust/src/bootstrap/bootstrap.py"],
    }

    from model.pico_type.realdata import fetch_raw
    files = known_files.get(language, [])
    for url in files:
        if len(samples) >= samples_per_lang:
            break
        content = fetch_raw(url, token)
        if content and MIN_BYTES <= len(content) <= MAX_BYTES:
            samples.append(Sample(
                data=content[:MAX_BYTES],
                coarse=_COARSE["code"],
                modality=_MODALITY["textual"],
                code_lang=lang_idx,
            ))

    print(f"  +{len(samples) - n_search} from known files = {len(samples)}")
    return samples[:samples_per_lang]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="", help="GitHub token")
    parser.add_argument("--samples-per-lang", type=int, default=20)
    parser.add_argument("--output", default="checkpoints/real_samples.pkl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_samples = []
    completed_langs = set()

    if args.resume and os.path.exists(output_path):
        with open(output_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                all_samples = data.get("samples", [])
                completed_langs = set(data.get("completed_langs", []))
            else:
                all_samples = data
        print(f"Resumed: {len(all_samples)} samples, {len(completed_langs)} languages")

    langs_to_do = [lang for lang in CODE_LANG_LABELS if lang not in completed_langs]
    print(f"Fetching {len(langs_to_do)} remaining ({len(completed_langs)} done)")

    for i, lang in enumerate(langs_to_do):
        print(f"[{i+1}/{len(langs_to_do)}] {lang}...", end=" ", flush=True)
        t0 = time.time()
        samples = fetch_fast(lang, args.samples_per_lang, args.token)
        elapsed = time.time() - t0
        all_samples.extend(samples)
        completed_langs.add(lang)
        print(f"  ({elapsed:.0f}s, total={len(all_samples)})")

        with open(output_path, "wb") as f:
            pickle.dump({"samples": all_samples, "completed_langs": list(completed_langs)}, f)

        time.sleep(0.3)

    print(f"\nDone: {len(all_samples)} samples across {len(completed_langs)} languages")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
