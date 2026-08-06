"""Compare worker (Node) results against Python onnxruntime reference."""

import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)

import numpy as np
import onnxruntime as ort

import LABELS


def classify(text, tier):
    model_path = f"{REPO_ROOT}/dist/single_onnx/picotype_{tier}.onnx"
    session = ort.InferenceSession(model_path)
    text_bytes = text.encode("utf-8")[:1024]
    ids = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
    seq = len(ids)
    padded = np.zeros(1024, np.int64)
    padded[:seq] = ids
    mask = np.zeros(1024, np.bool_)
    mask[:seq] = True
    outs = session.run(None, {"input_ids": padded[None, :], "attention_mask": mask[None, :]})
    result = {}
    for name, logits in zip(LABELS.ALL_HEADS, outs):
        e = np.exp(logits[0] - logits[0].max())
        probs = e / e.sum()
        if name == "risk":
            result[name] = {LABELS.LABEL_TABLES[name][i]: float(probs[i]) for i in range(len(probs))}
        else:
            idx = int(np.argmax(probs))
            result[name] = {"label": LABELS.LABEL_TABLES[name][idx], "confidence": float(probs[idx])}
    return result


with open("/tmp/worker_results.json") as f:
    worker_results = json.load(f)
samples = [
    "def hello():\n    print('Hello, world!')",
    "The quick brown fox jumps over the lazy dog.",
    "<html><body><h1>Welcome</h1></body></html>",
    '{\n  "name": "pico-type",\n  "version": "0.2.0"\n}',
    "#!/usr/bin/env python3\nimport os\nprint('hello')",
    "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR",
    "sk-9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0",
    '{"a":1,"b":[true,null,2.5]}',
    "\u00a1Hola! \u00bfC\u00f3mo est\u00e1s? Estoy muy bien, gracias.",
    "SELECT * FROM users WHERE id = 42;",
]
fails = 0
for i, text in enumerate(samples):
    for tier in ["tiny", "small", "base", "pro"]:
        key = f"{i}|{tier}"
        r = worker_results.get(key)
        if r is None or r["status"] != 200:
            continue
        ref = classify(text, tier)
        for head in LABELS.ALL_HEADS:
            if head == "risk":
                for lbl, prob in ref[head].items():
                    w = r["body"][head].get(lbl, 0)
                    if abs(w - prob) > 0.002:
                        print(f"MISMATCH risk {key[:30]} {lbl}: worker={w:.4f} py={prob:.4f}")
                        fails += 1
            else:
                wl, wc = r["body"][head]["label"], r["body"][head]["confidence"]
                pl, pc = ref[head]["label"], ref[head]["confidence"]
                if wl != pl or abs(wc - pc) > 0.002:
                    print(f"MISMATCH {head} {key[:40]!r}: worker={wl}({wc:.4f}) py={pl}({pc:.4f})")
                    fails += 1
print("fails:", fails)
