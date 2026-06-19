"""pico-type Gradio Space: classify content type, language, and risk."""

from __future__ import annotations

import os

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
    "zh", "ja", "ko", "vi", "th", "id", "hi",
]
FILE_MIME_LABELS = [
    "application/pdf", "application/zip", "application/gzip", "application/x-tar",
    "application/x-7z-compressed", "application/x-rar-compressed", "application/x-bzip2",
    "application/x-xz", "application/json", "application/xml", "application/yaml",
    "application/octet-stream", "application/x-executable", "application/x-mach-binary",
    "application/x-elf", "application/x-deb", "application/x-rpm",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
    "application/msword", "application/rtf", "application/epub+zip",
    "application/x-ndjson",
    "text/plain", "text/csv", "text/html", "text/xml", "text/markdown",
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml",
    "image/bmp", "image/tiff", "image/heic", "image/heif", "image/avif",
    "image/x-icon", "image/vnd.adobe.photoshop",
    "video/mp4", "video/webm", "video/x-matroska", "video/quicktime",
    "video/x-msvideo", "video/x-flv", "video/x-mpeg",
    "audio/mpeg", "audio/ogg", "audio/wav", "audio/flac", "audio/aac",
    "audio/x-m4a", "audio/webm", "audio/midi",
    "font/ttf", "font/otf", "font/woff", "font/woff2",
    "application/x-sqlite3", "application/x-parquet",
    "application/x-protobuf", "application/x-flatbuffers",
    "application/x-cpio", "application/x-iso9660-image",
    "application/vnd.android.package-archive", "application/x-jar",
    "application/x-python-bytecode", "application/x-archive",
    "application/pgp-encrypted", "application/pgp-signature",
    "application/x-x509-ca-cert", "application/x-pem-file",
    "application/vnd.tcpdump.pcap",
    "application/java-vm",
    "application/x-matlab-data",
    "application/x-shockwave-flash",
    "application/x-font-ttf", "application/x-font-otf",
    "application/wasm", "application/x-ruby",
    "application/javascript", "application/ecmascript",
    "application/x-bittorrent", "application/x-dvi",
    "chemical/x-mdl-sdfile",
    "application/x-lzma",
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
ONNX_VERSION = "1"  # bump to force re-download (e.g., after model update)


def _ensure_onnx(tier: str):
    path = os.path.join(MODEL_DIR, f"picotype_{tier}.onnx")
    version_file = os.path.join(MODEL_DIR, ".onnx_version")
    cached_version = ""
    if os.path.exists(version_file):
        with open(version_file) as f:
            cached_version = f.read().strip()
    if not os.path.exists(path) or cached_version != ONNX_VERSION:
        from huggingface_hub import hf_hub_download
        for t in ["tiny", "small", "base", "pro"]:
            fname = f"picotype_{t}.onnx"
            hf_hub_download("eulogik/pico-type", filename=fname, local_dir=MODEL_DIR, force_download=True)
            hf_hub_download("eulogik/pico-type", filename=f"{fname}.data", local_dir=MODEL_DIR, force_download=True)
        with open(version_file, "w") as f:
            f.write(ONNX_VERSION)
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
    mask = np.zeros(1024, dtype=bool)
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
