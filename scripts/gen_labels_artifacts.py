#!/usr/bin/env python3
"""Generate LABELS.py (Space) and labels.js (Worker) from model/pico_type/labels.py."""

from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "model" / "pico_type" / "labels.py"


def extract() -> dict[str, list[str]]:
    tree = ast.parse(SRC.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.endswith("_LABELS") and isinstance(node.value, ast.List):
                out[name] = [e.value for e in node.value.elts]
    return out


def main() -> None:
    labels = extract()
    if len(labels) != 7:
        sys.exit(f"expected 7 label tables, got {len(labels)}: {sorted(labels)}")
    head_map = {name[:-7].lower(): values for name, values in labels.items()}

    py = REPO / "LABELS.py"
    js = REPO / "workers" / "src" / "labels.js"
    js.parent.mkdir(parents=True, exist_ok=True)

    py.write_text(
        '"""Label vocabularies, generated from model/pico_type/labels.py. Do not edit by hand."""\n\n'
        + "\n".join(f"{k} = {v!r}\n" for k, v in labels.items())
        + "\nLABEL_TABLES = {\n"
        + "\n".join(f'    "{head}": {name},' for name, head in zip(labels, head_map))
        + "\n}\n"
        + 'ALL_HEADS = ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk")\n'
    )
    js.write_text(
        "// Generated from model/pico_type/labels.py. Do not edit by hand.\n"
        'export const LABEL_TABLES = ' + repr(head_map) + ";\n"
        'export const ALL_HEADS = ["coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk"];\n'
    )
    print("wrote", py, "and", js)


if __name__ == "__main__":
    main()
