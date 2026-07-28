<div align="center">

# pico-type 🔍

**A tiny byte-level multi-head content classifier** — ~1.5M params, ~200KB ONNX, <12ms inference.

Classifies any content from raw bytes: **coarse type · modality · subtype · code language · text language · file MIME · risk flags**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![PyPI](https://img.shields.io/pypi/v/pico-type?color=blue)](https://pypi.org/project/pico-type/)
[![ONNX](https://img.shields.io/badge/ONNX-exported-success)]()
[![CI](https://github.com/eulogik/pico-type/actions/workflows/ci.yml/badge.svg)](https://github.com/eulogik/pico-type/actions/workflows/ci.yml)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/eulogik/pico-type)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/eulogik/pico-type)

</div>

---

## ✨ Features

- **No tokenizer** — operates directly on raw UTF-8 bytes (supports all languages, no preprocessing)
- **7 heads, one forward pass** — coarse type, modality, subtype, code language, text language, file MIME, risk flags
- **4 Matryoshka tiers** — tiny (16d) → small (64d) → base (192d) → pro (576d) — same trunk, accuracy scales with dim
- **~200KB ONNX** — deploy on edge devices, serverless, browser (WebAssembly/ONNX Runtime Web)
- **<12ms inference** on CPU via ONNX Runtime
- **CLI, Python API, Gradio Space, MCP server** — ready to use

## 📊 Evaluation

### Overall Accuracy (v2 — trained on real data)

| Head | Classes | Accuracy | Dataset |
|------|---------|----------|---------|
| coarse | 12 | **100%** | Synthetic eval |
| modality | 8 | **100%** | Synthetic eval |
| subtype | 24 | **93.8%** | Synthetic eval |
| **code_lang** | **62** | **60.3%** | [The Heap](https://huggingface.co/datasets/AISE-TUDelft/the-heap) — 24 real-world langs, 1,200 samples |
| **text_lang** | **30** | **98.3%** | [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) — 30 langs, 1,500 samples |
| file_mime | 90 | **100%** | Synthetic eval |
| risk (mAP) | 6 | **100%** | Synthetic eval |

*v0.1 baseline (synthetic-only): code_lang 3%, text_lang 19%. Real-data training in v2 improves code by **57pp** and text by **79pp**.*

### Code Language — Per-Language Accuracy

| Excellent (90%+) | Good (70–89%) | Needs Work (<50%) |
|---|---|---|
| cpp 96%, dart 98%, erlang 98%, rust 98%, r 94%, swift 92%, python 88%, lua 88% | go 86%, ruby 86%, ocaml 84%, php 78%, csharp 76%, java 76%, kotlin 76%, c 62% | perl 50%, haskell 24%, scala 4%, javascript 2%, clojure 0%, elixir 0%, julia 0%, sql 0% |

> **Note**: Low-accuracy languages have fewer real training samples. More data will improve them.

## 🚀 Quick Start

### Install

```bash
pip install picotype
```

### CLI

```bash
# Classify from stdin
echo "def hello(name):\n    return f'Hi {name}'" | picotype --pretty

# Classify a file
picotype --file document.txt

# Classify clipboard content
picotype --clip

# All 4 tiers available
echo "..." | picotype --tier pro
```

### Python API

```python
from picotype import load_onnx_model, run_onnx

session = load_onnx_model("base")
result = run_onnx(session, "def hello(): pass")
print(result)
# {
#   "coarse": "code",
#   "code_language": "python",
#   "modality": "textual",
#   "confidence": 0.98,
#   ...
# }
```

### MCP Server (for Claude Desktop, Cursor, etc.)

```bash
pip install picotype
PICOTYPE_MODEL_DIR=./checkpoints python -m model.pico_type.mcp_server
```

Then add to your MCP config:
```json
{
  "mcpServers": {
    "pico-type": {
      "command": "python",
      "args": ["-m", "model.pico_type.mcp_server"],
      "env": { "PICOTYPE_MODEL_DIR": "./checkpoints" }
    }
  }
}
```

### Gradio Web UI

Try it live: [huggingface.co/spaces/eulogik/pico-type](https://huggingface.co/spaces/eulogik/pico-type)

## 🏗 Architecture

```
Bytes ─▶ ByteEmbed(256→96d) ─▶ 3×Conv1D(k=3,5,7) ─▶ 2×BiAttention(RoPE) ─▶ Pool ─▶ 7×Matryoshka Heads
```

| Component | Detail |
|-----------|--------|
| **ByteEmbed** | Lookup-free embedding — each byte value (0–255) maps to a learned 96-dim vector |
| **Conv1D** | 3 parallel depthwise convolutions (kernel widths 3, 5, 7) with residual + layer norm |
| **BiAttention** | Bidirectional self-attention with Rotary Position Embeddings (RoPE), 4 heads |
| **Pool** | Mean + max + std deviation concatenation → fixed-size representation |
| **Heads** | Matryoshka-style: slice pool dim to 16/64/192/576, project to 7 linear classifiers |

Total parameters: **1.43M** (tiny) / **1.45M** (small) / **1.48M** (base) / **1.56M** (pro)

## 🔧 Model Tiers

| Tier | Dim | Params | ONNX Size | Accuracy Multiplier |
|------|-----|--------|-----------|-------------------|
| tiny | 16 | 1.43M | 203 KB | 0.65× |
| small | 64 | 1.45M | 203 KB | 0.82× |
| base | 192 | 1.48M | 206 KB | 1.0× (reference) |
| pro | 576 | 1.56M | 202 KB | 1.05× |

All tiers share the same backbone; only the final linear projection layers differ. Higher-tier models use more dimensions for finer-grained classification.

## 🧪 Classification Heads

| Head | Classes | What It Detects |
|------|---------|-----------------|
| **coarse** | 12 | text, code, link, image, file, config, markup, data, error, secret, archive, binary |
| **modality** | 8 | textual, binary_image, binary_archive, binary_executable, binary_document, etc. |
| **subtype** | 24 | json, yaml, toml, csv, html, markdown, sql, log, dockerfile, makefile, etc. |
| **code_lang** | **62** | python, javascript, typescript, java, c, cpp, go, rust, ruby, php, swift, kotlin, and 50 more |
| **text_lang** | 30 | en, es, fr, de, it, pt, nl, ru, zh, ja, ko, vi, th, id, and 15 more |
| **file_mime** | **90** | application/json, image/png, video/mp4, font/ttf, application/wasm, and 84 more |
| **risk** | 6 | api_key, jwt, password, email, phone, ssh_key |

## 🌐 Deployment

| Platform | Link | Notes |
|----------|------|-------|
| **HuggingFace Space** | [eulogik/pico-type](https://huggingface.co/spaces/eulogik/pico-type) | Gradio web UI, no GPU needed |
| **HuggingFace Model** | [eulogik/pico-type](https://huggingface.co/eulogik/pico-type) | ONNX models + export metadata |
| **GitHub** | [eulogik/pico-type](https://github.com/eulogik/pico-type) | Source code, training, paper |
| **PyPI** | `pip install picotype` | Python package |
| **ONNX Runtime** | Use with onnxruntime.js | Browser/Node.js deployment |

## 📚 Resources

- [Paper](paper/main.pdf) — Architecture, training, and evaluation details
- [Model Card](MODEL_CARD.md) — Detailed architecture and training configuration
- [Walkthrough](walkthrough.md) — Development log and decisions
- [Architecture Plan](docs/PLAN.md) — Original design document

## 📄 License

Apache 2.0

---

<div align="center">
<sub>Built with PyTorch · ONNX · Gradio · HuggingFace</sub>
</div>
