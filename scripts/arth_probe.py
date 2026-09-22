"""ARTH Wk 1-2 probe: trunk-freeze verification, param budget, latency smoke test.

Usage: python scripts/arth_probe.py [--out scripts/arth_baseline.json]
- Loads v0.2 best.pt (frozen trunk reference).
- Counts params per component (embed / conv / attn-per-layer / pool / heads).
- Measures ONNX 1Q P50 single-thread at 1024B (no-regression gate: base <= 18ms).
- Runs fixed parity probe set through ONNX base tier, dumps predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.cli import load_onnx_model, run_onnx

BEST_PT = os.path.join(ROOT, "checkpoints", "best.pt")
ONNX_DIR = os.path.join(ROOT, "checkpoints")

PROBE_SET: list[tuple[str, bytes]] = [
    ("py_code", b"def hello(name):\n    print(f'hi {name}')\n"),
    ("js_code", b"const x = () => { return [1,2,3].map(n => n*n); };"),
    ("en_text", b"The quick brown fox jumps over the lazy dog. " * 8),
    ("json", b'{"name": "pico", "params": 1500000, "tags": ["ml", "edge"]}'),
    ("yaml", b"model: pico-type\nversion: 0.2.2\ntiers: [tiny, small, base, pro]\n"),
    ("png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64),
    ("secret", b"aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"),
    ("error", b"Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: bad\n"),
    ("sql", b"SELECT id, name FROM users WHERE active = 1 ORDER BY name;\n"),
    ("markdown", b"# pico-type\n\nA tiny classifier. See [docs](https://eulogik.github.io/pico-type).\n"),
]


def count_params(mod: torch.nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def param_budget() -> dict:
    ckpt = torch.load(BEST_PT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = PicoType(PicoTypeConfig())
    model.load_state_dict(sd)
    model.eval()
    parts = {
        "embed": count_params(model.embed),
        "conv_total": count_params(model.conv_blocks),
        "attn_per_layer": [count_params(b) for b in model.attn_blocks],
        "pool": count_params(model.pool),
        "heads_total": count_params(model.heads),
        "total_all": count_params(model),
    }
    parts["trunk_frozen"] = parts["embed"] + parts["conv_total"] + sum(parts["attn_per_layer"]) + parts["pool"]
    for tier in ("tiny", "small", "base", "pro"):
        parts[f"tier_{tier}_total"] = model.parameter_count(tier)
    parts["ckpt_step"] = ckpt.get("step")
    parts["ckpt_rwa"] = ckpt.get("real_world_accuracy")
    return parts


def latency_probe(session, n_warm: int = 20, n_run: int = 100) -> dict:
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 256, size=1024, dtype=np.uint8))
    for _ in range(n_warm):
        run_onnx(session, data)
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        run_onnx(session, data)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"n": n_run, "p50_ms": ts[n_run // 2], "p95_ms": ts[int(n_run * 0.95)], "min_ms": ts[0], "max_ms": ts[-1]}


def parity_probe(session) -> dict:
    out = {}
    for name, data in PROBE_SET:
        r = run_onnx(session, data)
        out[name] = {h: r[h].get("label") for h in r if h != "risk"}
        out[name]["risk_top"] = max(r["risk"], key=r["risk"].get)
    return out


def torch_parity_probe(best_pt: str) -> dict:
    from model.pico_type.cli import run_torch

    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = PicoType(PicoTypeConfig())
    model.load_state_dict(sd)
    model.eval()
    out = {}
    for name, data in PROBE_SET:
        r = run_torch((model, "base"), "base", data)
        out[name] = {h: r[h].get("label") for h in r if h != "risk"}
        out[name]["risk_top"] = max(r["risk"], key=r["risk"].get)
    return out


def machine_info() -> dict:
    import platform

    return {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "arth_baseline.json"))
    args = ap.parse_args()

    if not os.path.exists(BEST_PT):
        sys.exit(f"missing {BEST_PT}")
    budget = param_budget()
    print("== param budget (v0.2 trunk) ==")
    for k, v in budget.items():
        print(f"  {k}: {v}")

    mach = machine_info()
    print("== machine ==", mach)

    session_st = load_onnx_model("base", ONNX_DIR, single_thread=True)
    lat_st = latency_probe(session_st)
    print("== latency 1Q 1024B single-thread (base) ==")
    print(f"  p50={lat_st['p50_ms']:.2f}ms p95={lat_st['p95_ms']:.2f}ms")

    session_mt = load_onnx_model("base", ONNX_DIR, single_thread=False)
    lat_mt = latency_probe(session_mt)
    print("== latency 1Q 1024B default-threads (base, README conditions) ==")
    print(f"  p50={lat_mt['p50_ms']:.2f}ms p95={lat_mt['p95_ms']:.2f}ms")
    gate = "PASS" if lat_mt["p50_ms"] <= 18.0 else "FAIL(>18ms)"
    print(f"  gate base<=18ms (default threads): {gate}")

    parity = parity_probe(session_mt)
    print("== parity probe (ONNX base) ==")
    for name, r in parity.items():
        print(f"  {name}: coarse={r.get('coarse')} code={r.get('code_lang')} text={r.get('text_lang')}")

    torch_parity = torch_parity_probe(BEST_PT)
    print("== parity probe (torch best.pt base) ==")
    for name, r in torch_parity.items():
        print(f"  {name}: coarse={r.get('coarse')} code={r.get('code_lang')} text={r.get('text_lang')}")

    with open(args.out, "w") as f:
        json.dump(
            {"machine": mach, "params": budget, "latency_single": lat_st, "latency_multi": lat_mt,
             "parity_onnx": parity, "parity_torch": torch_parity},
            f, indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
