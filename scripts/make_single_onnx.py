#!/usr/bin/env python3
"""Merge external ONNX weights into single-file models and lower IR version to 8.

IR 8 is the highest supported by onnxruntime-web 1.14-1.16 (WASM builds used on
Cloudflare Workers); the exported models use IR 10+ which those builds reject.
The IR version is metadata only — opset (18) and weights are unchanged.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys

import onnx
from onnx.external_data_helper import load_external_data_for_model

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "onnx_models_v2"
DST_DIR = REPO / "dist" / "single_onnx"
TARGET_IR = 8


def main() -> None:
    tiers = ["tiny", "small", "base", "pro"]
    shutil.rmtree(DST_DIR, ignore_errors=True)
    DST_DIR.mkdir(parents=True)
    for tier in tiers:
        src = SRC_DIR / f"picotype_{tier}.onnx"
        dst = DST_DIR / f"picotype_{tier}.onnx"
        model = onnx.load(str(src), load_external_data=False)
        load_external_data_for_model(model, str(SRC_DIR))
        model.ir_version = TARGET_IR
        onnx.save_model(model, str(dst), save_as_external_data=False)
        size_mb = os.path.getsize(dst) / 1e6
        print(f"{tier}: single-file IR={model.ir_version} opset={model.opset_import[0].version} {size_mb:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
