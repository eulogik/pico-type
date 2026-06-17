"""pico-type Gradio Space: classify content type, language, and risk."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np

ALL_HEADS = ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk")

COARSE_LABELS = [
    "text", "code", "link", "image", "file", "config",
    "markup", "data", "error", "secret", "archive", "binary",
]
MODALITY_LABELS = [
    "textual", "binary_image", "binary_archive", "binary_executable",
    "binary_document", "binary_audio", "binary_video", "binary_other",
]
SUBTYPE_LABELS = [
    "json", "yaml", "toml", "ini", "csv", "tsv", "xml",
    "html", "markdown", "rst", "asciidoc", "tex",
    "sql", "graphql", "protobuf", "msgpack",
    "log", "diff", "patch", "env",
    "shell", "makefile", "dockerfile", "gitignore",
]
CODE_LANG_LABELS = [
    "python", "javascript", "typescript", "jsx", "tsx",
    "java", "kotlin", "scala", "groovy", "clojure",
    "c", "cpp", "csharp", "fsharp", "objectivec",
    "go", "rust", "zig",
    "ruby", "php", "perl", "lua", "tcl",
    "swift", "dart", "julia", "nim", "crystal",
    "haskell", "ocaml", "elm", "erlang", "elixir",
    "lisp", "scheme", "racket",
    "r", "matlab", "octave", "sas", "stata",
    "sql", "plsql", "tsql",
    "html", "css", "scss", "sass", "less",
    "bash", "zsh", "fish", "powershell",
    "vim", "fortran", "cobol", "ada", "pascal",
    "delphi", "vb", "prolog", "vhdl",
]
TEXT_LANG_LABELS = [
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da",
    "fi", "pl", "cs", "sk", "hu", "ro", "el", "tr",
    "ru", "uk", "bg", "sr", "hr",
    "zh", "ja", "ko", "ar", "hi", "th", "vi",
]
FILE_MIME_LABELS = [
    "text/html", "application/json", "application/xml", "text/yaml",
    "text/toml", "text/ini", "text/csv", "text/tsv", "text/markdown",
    "text/plain", "text/x-python", "text/x-java", "text/x-c",
    "text/x-cpp", "text/x-rust", "text/x-go", "text/x-ruby",
    "text/x-php", "text/x-javascript", "text/x-typescript",
    "text/x-shellscript", "text/x-sql", "text/x-dockerfile",
    "text/x-makefile", "text/x-yaml", "text/x-diff", "text/x-log",
    "text/x-env", "text/x-tex", "text/x-asciidoc", "text/x-rst",
    "application/pdf", "application/zip", "application/gzip",
    "application/x-tar", "application/x-7z-compressed", "application/x-rar-compressed",
    "application/x-bzip2", "application/x-xz", "application/x-iso9660-image",
    "application/vnd.sqlite3", "application/x-parquet",
    "application/x-elf", "application/x-mach-binary",
    "application/x-pe-executable", "application/java-archive",
    "application/wasm", "application/vnd.debian.binary-package",
    "application/x-apple-diskimage", "application/x-msdownload",
    "application/x-sharedlib", "application/x-object",
    "application/x-pcap", "application/x-hdf5", "application/x-netcdf",
    "application/xml", "application/atom+xml", "application/rss+xml",
    "application/rdf+xml", "application/xhtml+xml",
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp",
    "image/tiff", "image/svg+xml", "image/x-icon", "image/avif",
    "audio/mpeg", "audio/wav", "audio/ogg", "audio/flac",
    "audio/aac", "audio/mp4", "audio/webm",
    "video/mp4", "video/webm", "video/ogg", "video/x-msvideo",
    "video/quicktime", "video/x-matroska",
    "font/ttf", "font/otf", "font/woff", "font/woff2",
    "application/octet-stream", "application/unknown",
]
RISK_LABELS = ["api_key", "jwt", "ssh_key", "password", "email", "phone"]

LABEL_TABLES = {
    "coarse": COARSE_LABELS,
    "modality": MODALITY_LABELS,
    "subtype": SUBTYPE_LABELS,
    "code_lang": CODE_LANG_LABELS,
    "text_lang": TEXT_LANG_LABELS,
    "file_mime": FILE_MIME_LABELS,
    "risk": RISK_LABELS,
}

MODEL_DIR = "."


def _ensure_onnx(tier: str):
    path = os.path.join(MODEL_DIR, f"picotype_{tier}.onnx")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        for t in ["tiny", "small", "base", "pro"]:
            fname = f"picotype_{t}.onnx"
            hf_hub_download("eulogik/pico-type", filename=fname, local_dir=MODEL_DIR)
            hf_hub_download("eulogik/pico-type", filename=f"{fname}.data", local_dir=MODEL_DIR)
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
