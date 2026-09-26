"""pico-type Gradio Space: classify content type, language, and risk.

Ships two models:
  - ARTH V2 (default): arth_full_base.onnx + 14-label Risk++ with per-label
    fitted thresholds and fired flags (huggingface.co/eulogik/pico-type).
  - legacy v02 tiers (tiny/small/base/pro) from eulogik/pico-type-v02.
"""

from __future__ import annotations

import json
import os

import gradio as gr
import numpy as np

from LABELS import ALL_HEADS, LABEL_TABLES

MODEL_DIR = "."
ARTH_REPO = "eulogik/pico-type"
ARTH_ONNX = "arth_full_base.onnx"
ARTH_THRESHOLDS = "risk_thresholds.json"
# MUST match the riskpp_logits output column order (model/pico_type/arth.py
# RISK_PLUS_LABELS) — the thresholds JSON is name-keyed and its key order
# differs, so probs are zipped against this list, never against JSON order.
RISKPP_LABELS = [
    "api_key", "jwt", "ssh_key", "password", "email", "phone",
    "prompt_injection", "jailbreak", "pii_ssn", "pii_card",
    "secrets_aws", "secrets_github", "sql_injection", "xss_payload",
]


def _ensure_onnx(tier: str):
    path = os.path.join(MODEL_DIR, f"picotype_{tier}.onnx")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        for t in ["tiny", "small", "base", "pro"]:
            hf_hub_download("eulogik/pico-type-v02", filename=f"picotype_{t}.onnx", local_dir=MODEL_DIR)
    return path


def _load_session(tier: str):
    import onnxruntime as ort
    path = _ensure_onnx(tier)
    return ort.InferenceSession(path)


SESSIONS = {}


def _get_session(tier: str):
    if tier not in SESSIONS:
        SESSIONS[tier] = _load_session(tier)
    return SESSIONS[tier]


ARTH = {}


def _get_arth():
    """Lazy-load the ARTH V2 ONNX + 14-label thresholds from the model repo."""
    if not ARTH:
        from huggingface_hub import hf_hub_download
        onnx_path = hf_hub_download(ARTH_REPO, ARTH_ONNX, local_dir=MODEL_DIR)
        th_path = hf_hub_download(ARTH_REPO, ARTH_THRESHOLDS, local_dir=MODEL_DIR)
        with open(th_path) as f:
            thresholds = {k: float(v) for k, v in json.load(f).items()}
        if set(thresholds) != set(RISKPP_LABELS):
            raise ValueError(f"threshold labels {sorted(thresholds)} != {sorted(RISKPP_LABELS)}")
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        n_struct = int(sess.get_inputs()[3].shape[1])
        ARTH.update(sess=sess, thresholds=thresholds, n_struct=n_struct)
    return ARTH


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def classify(text: str, tier: str) -> dict:
    if not text.strip():
        return {}
    session = _get_session(tier)
    text_bytes = text.encode("utf-8")[:1024]
    ids = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
    seq_len = len(ids)
    padded = np.zeros(1024, dtype=np.int64)
    padded[:seq_len] = ids
    mask = np.zeros(1024, dtype=np.bool_)
    mask[:seq_len] = True
    outs = session.run(None, {"input_ids": padded[None, :], "attention_mask": mask[None, :]})
    result = {}
    for name, logits in zip(ALL_HEADS, outs):
        probs = _softmax(logits[0])
        if name == "risk":
            result[name] = {LABEL_TABLES[name][i]: float(probs[i]) for i in range(len(probs))}
        else:
            idx = int(np.argmax(probs))
            result[name] = {"label": LABEL_TABLES[name][idx], "confidence": float(probs[idx])}
    return result


def classify_arth(text: str) -> tuple[dict, dict]:
    """ARTH V2: legacy 7 heads + Risk++ 14 labels (sigmoid, per-label thresholds)."""
    arth = _get_arth()
    if not text.strip():
        return {}, {}
    text_bytes = text.encode("utf-8")[:1024]
    ids = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
    seq_len = len(ids)
    padded = np.zeros(1024, dtype=np.int64)
    padded[:seq_len] = ids
    mask = np.zeros(1024, dtype=np.bool_)
    mask[:seq_len] = True
    outs = arth["sess"].run(
        None,
        {
            "input_ids": padded[None, :],
            "attention_mask": mask[None, :],
            "opt_embs": np.zeros((1, 4, 96), dtype=np.float32),
            "struct_feats": np.zeros((1, arth["n_struct"]), dtype=np.float32),
            "act_stats": np.zeros((1, 4), dtype=np.float32),
        },
    )
    result = {}
    for name, logits in zip(ALL_HEADS, outs[: len(ALL_HEADS)]):
        probs = _softmax(logits[0])
        if name == "risk":
            result[name] = {LABEL_TABLES[name][i]: float(probs[i]) for i in range(len(probs))}
        else:
            idx = int(np.argmax(probs))
            result[name] = {"label": LABEL_TABLES[name][idx], "confidence": float(probs[idx])}
    rpp_probs = 1.0 / (1.0 + np.exp(-outs[9][0]))
    if len(rpp_probs) != len(RISKPP_LABELS):
        raise ValueError(f"riskpp output {len(rpp_probs)} != {len(RISKPP_LABELS)} labels")
    rpp = {}
    for label, prob in zip(RISKPP_LABELS, rpp_probs):
        fired = bool(prob >= arth["thresholds"][label])
        rpp[f"{label} {'[FLAG]' if fired else ''}".strip()] = float(round(float(prob), 4))
    return result, rpp


def build_ui():
    with gr.Blocks(title="pico-type", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # pico-type 🔍
            A tiny byte-level multi-head content classifier (~1.5M params).
            **ARTH V2 (default)** adds 14-label **Risk++** with per-label fitted thresholds — `[FLAG]` marks fired risks.
            [Paper (arXiv:2608.14658)](https://arxiv.org/abs/2608.14658)
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Content",
                    placeholder="Paste or type content to classify...",
                    lines=10,
                )
                with gr.Row():
                    tier_selector = gr.Radio(
                        choices=["arth", "tiny", "small", "base", "pro"],
                        value="arth",
                        label="Model",
                    )
                    submit_btn = gr.Button("Classify", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear")

                gr.Examples(
                    examples=[
                        ["def hello():\n    print('Hello, world!')"],
                        ["The quick brown fox jumps over the lazy dog."],
                        ["<html><body><h1>Welcome</h1></body></html>"],
                        ["#!/usr/bin/env python3\nimport os\nprint('hello')"],
                        ["{\n  \"name\": \"pico-type\",\n  \"version\": \"0.1.0\"\n}"],
                        ["BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR"],
                        ["aws_access_key_id = AKIAIOSFODNN7EXAMPLE"],
                    ],
                    inputs=[text_input],
                    label="Try these examples",
                )

            with gr.Column(scale=2):
                output_labels = []
                with gr.Tabs():
                    for head_name in ALL_HEADS:
                        with gr.Tab(head_name.replace("_", " ").title()):
                            lbl = gr.Label(
                                value={},
                                label=head_name.replace("_", " ").title(),
                            )
                            output_labels.append(lbl)
                    with gr.Tab("Risk++ (14)"):
                        rpp_lbl = gr.Label(value={}, label="Risk++ (ARTH V2)")
                        output_labels.append(rpp_lbl)

        def handle_classify(text, tier):
            if tier == "arth":
                result, rpp = classify_arth(text)
            else:
                result, rpp = classify(text, tier), {}
            outputs = {}
            for head in ALL_HEADS:
                if head == "risk":
                    outputs[head] = result.get(head, {})
                else:
                    outputs[head] = {result.get(head, {}).get("label", "unknown"): result.get(head, {}).get("confidence", 0)}
            return [outputs[h] for h in ALL_HEADS] + [rpp]

        submit_btn.click(
            fn=handle_classify,
            inputs=[text_input, tier_selector],
            outputs=output_labels,
        )

        clear_btn.click(
            fn=lambda: (""),
            inputs=[],
            outputs=[text_input],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
