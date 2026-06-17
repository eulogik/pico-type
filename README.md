<div align="center">

# pico-type 🔍

**A tiny byte-level multi-head content classifier** — ~1.5M params, ~200KB ONNX, <12ms inference.

Classifies any content into **7 categories** from raw bytes: coarse type, modality, subtype, code language, text language, file MIME, and risk flags.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![ONNX](https://img.shields.io/badge/ONNX-exported-success)](checkpoints)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/eulogik/pico-type)
[![GitHub](https://img.shields.io/badge/GitHub-eulogik/pico--type-181717?logo=github)](https://github.com/eulogik/pico-type)

</div>

---

## ✨ Features

- **No tokenizer** — operates directly on raw UTF-8 bytes (supports all languages)
- **7 heads, one forward pass** — coarse type, modality, subtype, code lang, text lang, file MIME, risk
- **4 Matryoshka tiers** — tiny (16d) → small (64d) → base (192d) → pro (576d)
- **~200KB ONNX** — deploy on edge devices, serverless, browser (WebAssembly)
- **<12ms inference** on CPU via ONNX Runtime
- **CLI, Gradio Space, MCP server** — ready to use

## 📊 Performance

| Head | Classes | Accuracy |
|------|---------|----------|
| coarse | 12 | **99.4%** |
| modality | 8 | **100%** |
| subtype | 24 | **95.4%** |
| text_lang | 30 | **81.3%** |
| file_mime | 90 | **100%** |
| risk (mAP) | 6 | **90.5%** |

## 🚀 Quick Start

### CLI
```bash
pip install picotype

echo "def hello():\n    return 42" | picotype --pretty
picotype --file document.txt
picotype --clip
```

### Python
```python
from model.pico_type.cli import load_onnx_model, run_onnx

session = load_onnx_model("base", "checkpoints")
result = run_onnx(session, "def hello(): pass")
print(result)
```

### MCP Server (Claude/Cursor)
```bash
PICOTYPE_MODEL_DIR=./checkpoints python -m model.pico_type.mcp_server
```

## 🏗 Architecture

```
Bytes → ByteEmbed(256→96d) → 3×Conv1D(k=3,5,7) → 2×BiAttention(RoPE) → Pool(mean‖max‖std) → 7×Matryoshka Heads
```

- **ByteEmbed**: lookup-free byte embedding (256 vocab, 96 dim)
- **Conv1D**: 3 parallel kernels (width 3, 5, 7) with residual + layer norm
- **BiAttention**: bidirectional self-attention with RoPE, 4 heads, 96 dim
- **Pool**: mean + max + std concatenation
- **Matryoshka Heads**: 4 slices of the pooled vector (16/64/192/576 dim) → 7 linear classifiers

Total parameters: **1.43M** (tiny) / **1.45M** (small) / **1.48M** (base) / **1.56M** (pro)

## 🔧 Model Tiers

| Tier | Dim | Params | ONNX Size |
|------|-----|--------|-----------|
| tiny | 16 | 1.43M | 203 KB |
| small | 64 | 1.45M | 203 KB |
| base | 192 | 1.48M | 206 KB |
| pro | 576 | 1.56M | 202 KB |

All tiers share the same trunk; only the final linear layers differ.

## 🧪 Classification Heads

| Head | Classes | Examples |
|------|---------|----------|
| **coarse** | 12 | text, code, link, image, file, config, markup, data, error, secret, archive, binary |
| **modality** | 8 | textual, binary_image, binary_archive, binary_executable, etc. |
| **subtype** | 24 | json, yaml, toml, csv, html, markdown, sql, log, dockerfile, etc. |
| **code_lang** | 62 | python, javascript, typescript, java, c, cpp, go, rust, etc. |
| **text_lang** | 30 | en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, etc. |
| **file_mime** | 90 | text/html, application/json, application/pdf, image/png, video/mp4, etc. |
| **risk** | 6 | api_key, jwt, password, email, phone, ssh_key |

## 🌐 Deployment

| Platform | Location |
|----------|----------|
| HuggingFace Space | [eulogik/pico-type](https://huggingface.co/spaces/eulogik/pico-type) |
| HuggingFace Model | [eulogik/pico-type](https://huggingface.co/eulogik/pico-type) |
| GitHub | [eulogik/pico-type](https://github.com/eulogik/pico-type) |
| PyPI | `pip install picotype` |

## 📚 Documentation

- [Model Card](MODEL_CARD.md) — detailed architecture, training, and evaluation
- [Architecture Plan](docs/PLAN.md) — full design document
- [Walkthrough](walkthrough.md) — development log

## 📄 License

Apache 2.0
