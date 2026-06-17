"""pico-type CLI: classify content from stdin/file/clipboard."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .labels import ALL_HEADS, COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS, CODE_LANG_LABELS, TEXT_LANG_LABELS, FILE_MIME_LABELS, RISK_LABELS

LABEL_TABLES = {
    "coarse": COARSE_LABELS,
    "modality": MODALITY_LABELS,
    "subtype": SUBTYPE_LABELS,
    "code_lang": CODE_LANG_LABELS,
    "text_lang": TEXT_LANG_LABELS,
    "file_mime": FILE_MIME_LABELS,
    "risk": RISK_LABELS,
}


def load_onnx_model(tier: str = "base", model_dir: str = "checkpoints"):
    import onnxruntime as ort
    path = os.path.join(model_dir, f"picotype_{tier}.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    session = ort.InferenceSession(path)
    return session


def load_torch_model(tier: str = "base", checkpoint: str = ""):
    import torch
    from .arch import PicoType, PicoTypeConfig
    cfg = PicoTypeConfig(max_bytes=1024)
    model = PicoType(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, tier


def run_onnx(session, text: str, max_bytes: int = 1024) -> dict:
    text_bytes = text.encode("utf-8")[:max_bytes]
    ids = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
    seq_len = len(ids)
    if seq_len > max_bytes:
        ids = ids[:max_bytes]
        seq_len = max_bytes
    padded = np.zeros(max_bytes, dtype=np.int64)
    padded[:seq_len] = ids
    mask = np.zeros(max_bytes, dtype=np.bool_)
    mask[:seq_len] = True

    feed = {
        "input_ids": padded[None, :],
        "attention_mask": mask[None, :],
    }
    outs = session.run(None, feed)
    result = {}
    for name, logits in zip(ALL_HEADS, outs):
        probs = _softmax(logits[0])
        if name == "risk":
            result[name] = {LABEL_TABLES[name][i]: float(probs[i]) for i in range(len(probs))}
        else:
            idx = int(np.argmax(probs))
            label = LABEL_TABLES[name][idx]
            result[name] = {"label": label, "confidence": float(probs[idx]), "index": idx}
    return result


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_torch(model, tier: str, text: str, max_bytes: int = 1024) -> dict:
    import torch
    model = model[0] if isinstance(model, tuple) else model
    text_bytes = text.encode("utf-8")[:max_bytes]
    ids = torch.tensor([list(text_bytes)], dtype=torch.long)
    mask = torch.ones(1, ids.shape[1], dtype=torch.bool)
    with torch.no_grad():
        logits_dict = model(ids, mask)
    out = {}
    for head in ALL_HEADS:
        logits = logits_dict[head]
        tier_logits = logits[tier] if isinstance(logits, dict) else logits
        if isinstance(tier_logits, dict):
            tier_logits = tier_logits[tier]
        probs = torch.softmax(tier_logits[0], dim=-1)
        if head == "risk":
            out[head] = {LABEL_TABLES[head][i]: float(probs[i]) for i in range(len(probs))}
        else:
            idx = int(torch.argmax(probs).item())
            out[head] = {"label": LABEL_TABLES[head][idx], "confidence": float(probs[idx]), "index": idx}
    return out


def read_text(args) -> str:
    if args.text:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if args.clip:
        import subprocess
        return subprocess.check_output(["pbpaste"], text=True)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("No input provided. Use --text, --file, --clip, or pipe content.")


def build_parser():
    p = argparse.ArgumentParser(prog="picotype", description="Classify content type and risk")
    p.add_argument("--text", "-t", help="Text string to classify")
    p.add_argument("--file", "-f", help="File path to classify")
    p.add_argument("--clip", "-c", action="store_true", help="Classify clipboard content")
    p.add_argument("--tier", default="base", choices=["tiny", "small", "base", "pro"], help="Model tier")
    p.add_argument("--model-dir", default="checkpoints", help="Directory with ONNX models")
    p.add_argument("--checkpoint", help="PyTorch checkpoint (fallback if no ONNX)")
    p.add_argument("--pretty", "-p", action="store_true", help="Pretty-print JSON output")
    return p


def main():
    args = build_parser().parse_args()
    try:
        text = read_text(args)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print('{"error": "empty input"}')
        sys.exit(0)

    onnx_path = os.path.join(args.model_dir, f"picotype_{args.tier}.onnx")
    if os.path.exists(onnx_path):
        session = load_onnx_model(args.tier, args.model_dir)
        result = run_onnx(session, text)
    elif args.checkpoint:
        model = load_torch_model(args.tier, args.checkpoint)
        result = run_torch(model, args.tier, text)
    else:
        print(f"ONNX model not found at {onnx_path}. Use --checkpoint to use PyTorch.", file=sys.stderr)
        sys.exit(1)

    result["text_length"] = len(text)
    result["tier"] = args.tier
    indent = 2 if args.pretty else None
    json.dump(result, sys.stdout, indent=indent, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
