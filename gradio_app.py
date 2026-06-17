"""pico-type Gradio Space: classify content type, language, and risk."""

from __future__ import annotations

import os

import gradio as gr
import numpy as np

from model.pico_type.labels import ALL_HEADS, COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS, CODE_LANG_LABELS, TEXT_LANG_LABELS, FILE_MIME_LABELS, RISK_LABELS

LABEL_TABLES = {
    "coarse": COARSE_LABELS,
    "modality": MODALITY_LABELS,
    "subtype": SUBTYPE_LABELS,
    "code_lang": CODE_LANG_LABELS,
    "text_lang": TEXT_LANG_LABELS,
    "file_mime": FILE_MIME_LABELS,
    "risk": RISK_LABELS,
}

MODEL_DIR = os.environ.get("PICOTYPE_MODEL_DIR", "checkpoints")


def _load_session(tier: str):
    import onnxruntime as ort
    path = os.path.join(MODEL_DIR, f"picotype_{tier}.onnx")
    return ort.InferenceSession(path)


SESSIONS = {}


def _get_session(tier: str):
    if tier not in SESSIONS:
        SESSIONS[tier] = _load_session(tier)
    return SESSIONS[tier]


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


def build_ui():
    with gr.Blocks(title="pico-type", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # pico-type 🔍
            A tiny byte-level multi-head content classifier (~1.5M params).  
            Classifies content into **7 categories**: coarse type, modality, subtype, code language, text language, file MIME, and risk flags.
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
                        choices=["tiny", "small", "base", "pro"],
                        value="base",
                        label="Model Tier",
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

        def handle_classify(text, tier):
            result = classify(text, tier)
            outputs = {}
            for head in ALL_HEADS:
                if head == "risk":
                    outputs[head] = result.get(head, {})
                else:
                    outputs[head] = {result.get(head, {}).get("label", "unknown"): result.get(head, {}).get("confidence", 0)}
            return [outputs[h] for h in ALL_HEADS]

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
