---
license: apache-2.0
language:
- en
- multilingual
tags:
- byte-level
- content-classification
- onnx
- edge-ai
- matryoshka
- multi-head
- classifier
- clipboard
pipeline_tag: text-classification
library_name: pico-type
inference:
  parameters:
    provider: CPUExecutionProvider
---

<div align="center">

# pico-type 🔍

**A tiny byte-level multi-head content classifier** — ~1.5M params, ~209KB ONNX, <6ms inference.

_Classifies any content into **7 categories** from raw bytes in a single forward pass._

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![ONNX](https://img.shields.io/badge/ONNX-exported-success)](https://huggingface.co/eulogik/pico-type)
[![PyPI](https://img.shields.io/pypi/v/pico-type?color=blue)](https://pypi.org/project/pico-type/)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/eulogik/pico-type)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/eulogik/pico-type)
[![GitHub](https://img.shields.io/badge/GitHub-eulogik/pico--type-181717?logo=github)](https://github.com/eulogik/pico-type)
[![CI](https://img.shields.io/github/actions/workflow/status/eulogik/pico-type/ci.yml?logo=githubactions&label=CI)](https://github.com/eulogik/pico-type/actions)
[![DOI](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)

_Built by [**eulogik**](https://eulogik.com) — AI infrastructure for developers._

</div>

---

## ✨ Features

- **No tokenizer** — operates directly on raw UTF-8 bytes (supports all languages, zero pre-processing)
- **7 heads, one forward pass** — coarse type, modality, subtype, code lang, text lang, file MIME, risk flags
- **4 Matryoshka tiers** — tiny (16d) → small (64d) → base (192d) → pro (576d)
- **~200KB ONNX** — deploy on edge devices, serverless functions, browser (WebAssembly)
- **<6ms inference** on CPU via ONNX Runtime (base tier, 1024 bytes)
- **CLI, Gradio Space, MCP server** — ready for any integration

## 📊 Performance

| Head | Classes | Synthetic Accuracy | Real-World Accuracy |
|------|---------|-------------------|---------------------|
| coarse | 12 | **100%** | **86%** |
| modality | 8 | **100%** | **100%** |
| subtype | 24 | **95%** | **—** |
| code_lang | 62 | **39%** | **—** |
| text_lang | 30 | **99%** | **100%** |
| file_mime | 90 | **100%** | **—** |
| risk (mAP) | 6 | **100%** | **—** |

_Evaluated on 1000 synthetic samples + 21 hand-curated real-world inputs. Base tier, ~5ms inference._

> Note: code_lang synthetic accuracy reflects the challenge of 62-way classification with limited per-class support. Real-world accuracy across all heads is **52%** (11/21 correct), up from **23%** baseline before diverse training.

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
from picotype import PicoType, PicoTypeConfig, decode_output

model = PicoType(PicoTypeConfig()).eval()
# ... load checkpoint ...
result = decode_output(model(b"input bytes"), tier="base")
```

### MCP Server (Claude/Cursor)
```bash
PICOTYPE_MODEL_DIR=./checkpoints python -m model.pico_type.mcp_server
```

## 🏗 Architecture

```
Bytes → ByteEmbed(256→96d) → 3×Conv1D(k=3,5,7) → 2×BiAttention(RoPE) → Pool(mean‖max‖std) → 7×Matryoshka Heads
```

| Component | Description |
|-----------|-------------|
| **ByteEmbed** | `nn.Embedding(256, 96)` — lookup-free byte embedding |
| **Conv1D** | 3 parallel kernels (width 3, 5, 7) with residual + LayerNorm + GELU |
| **BiAttention** | Bidirectional self-attention with Rotary Position Embeddings, 4 heads |
| **Pool** | Mean + Max + Std concatenation over masked positions |
| **Matryoshka Heads** | 4 tier slices of the pooled vector → 7 linear classifiers |

**Total parameters**: 1.43M (tiny) / 1.45M (small) / 1.48M (base) / 1.56M (pro)

## 🔧 Model Tiers

| Tier | Dim | Params | ONNX Size | Speed |
|------|-----|--------|-----------|-------|
| tiny | 16 | 1.43M | 207 KB | ~3ms |
| small | 64 | 1.45M | 207 KB | ~4ms |
| base | 192 | 1.48M | 209 KB | ~5ms |
| pro | 576 | 1.56M | 206 KB | ~12ms |

All tiers share the same trunk; only the final linear layers differ. Switch tiers at inference with zero overhead.

## 🧪 Classification Heads

| Head | Classes | Gated By | Examples |
|------|---------|----------|----------|
| **coarse** | 12 | — | text, code, link, image, file, config, markup, data, error, secret, archive, binary |
| **modality** | 8 | — | textual, binary_image, binary_archive, binary_executable, binary_document, binary_audio, binary_video, binary_other |
| **subtype** | 24 | config, markup, data | json, yaml, toml, csv, html, markdown, sql, log, dockerfile |
| **code_lang** | 62 | code | python, javascript, typescript, java, c, cpp, go, rust, kotlin, swift, bash, sql |
| **text_lang** | 30 | text | en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi |
| **file_mime** | 90 | image, file | text/html, application/json, application/pdf, image/png, video/mp4 |
| **risk** | 6 | — | api_key, jwt, password, email, phone, ssh_key (probabilities) |

## 🌐 Deployment

| Platform | URL |
|----------|-----|
| HuggingFace Space | [eulogik/pico-type](https://huggingface.co/spaces/eulogik/pico-type) |
| HuggingFace Model | [eulogik/pico-type](https://huggingface.co/eulogik/pico-type) |
| GitHub | [eulogik/pico-type](https://github.com/eulogik/pico-type) |
| PyPI | `pip install picotype` |
| arXiv | [XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (coming soon) |

## 📚 Documentation

- [Model Card](MODEL_CARD.md) — detailed architecture, training, evaluation
- [Architecture Plan](docs/PLAN.md) — full design document
- [Walkthrough](walkthrough.md) — development log with all decisions

## 📄 License

Apache 2.0 — free for commercial and personal use.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://eulogik.com">eulogik</a></sub>
</div>
