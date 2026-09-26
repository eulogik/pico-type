<div align="center">

# pico-type 🔍

**A tiny byte-level multi-head content classifier** — ~1.5M params, ~9MB single-file ONNX (FP32), ~18ms CPU inference.

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
- **~9MB single-file ONNX (FP32)** — deploy on edge devices, serverless, browser (WebAssembly/ONNX Runtime Web)
- **~18ms inference** on CPU via ONNX Runtime
- **CLI, Python API, Gradio Space, MCP server** — ready to use

## 🛡️ ARTH V2 — Risk++ (new)

**[Try it live — pick the `arth` model](https://huggingface.co/spaces/eulogik/pico-type)** · [Model files](https://huggingface.co/eulogik/pico-type) (`arth_full_base.onnx`, `risk_thresholds.json`, `temperatures.json`, `arth_final_composite.pt`)

Frozen-trunk composite (ft3 student + retrained 14-label **Risk++** head) — audit recall up **5× with zero regressions** on every shipped gate:

| Metric | Before (ft3) | ARTH V2 (composite) |
|---|---|---|
| External-audit positive recall (held-out, never trained/fit on) | 0.107 | **0.536** (60/112) |
| Balanced error | 0.459 | **0.260** |
| Specificity | 0.961 | 0.944 |
| Labels with fitted thresholds | 9 | **14** |
| Legacy parity | 60/60 | 60/60 |
| Choice accuracy | 0.737 | 0.737 |
| Noul abstain | 0.929 | 0.929 |
| ToxicChat jailbreak recall / sel-acc@50 | 0.846 / 0.970 | **0.824 / 0.973** (gate PASS) |

Formerly-dead labels now detected: jwt / ssh_key / email **8/8**, password 0.875, phone 0.750. Artifact: **`arth_full_base.onnx` (11.66 MB, opset 18)** — auto-verified (torch-vs-ORT err 1.1e-05, parity 60/60, semantic 300/300, risk flags 11/11); INT8 4.28 MB experimental.

**Known limitations (measured):** pii_ssn 0.125, api_key 0.625, jailbreak 0.50 on embedded phrasing; 3/11 hand-picked benign probes still flag (incl. the accepted `print('hello world')`→sql); long-document signals dilute in fixed-window pooling; general-classification gates (AG/SST-2/Enron) stay at chance — this is a byte-pattern risk flagger, not a general text classifier.

### ARTH quickstart (ONNX)

```python
import json, math
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

sess = ort.InferenceSession(hf_hub_download("eulogik/pico-type", "arth_full_base.onnx"))
thrs = json.load(open(hf_hub_download("eulogik/pico-type", "risk_thresholds.json")))
LABELS = ["api_key","jwt","ssh_key","password","email","phone","prompt_injection",
          "jailbreak","pii_ssn","pii_card","secrets_aws","secrets_github",
          "sql_injection","xss_payload"]

raw = open("file.txt","rb").read()[:1024]
ids = np.zeros(1024, np.int64); ids[:len(raw)] = list(raw)
mask = np.zeros(1024, bool); mask[:len(raw)] = True
(logits,) = sess.run(["riskpp_logits"], {
    "input_ids": ids[None,:], "attention_mask": mask[None,:],
    "opt_embs": np.zeros((1,4,96), np.float32),
    "struct_feats": np.zeros((1,14), np.float32),
    "act_stats": np.zeros((1,4), np.float32)})
probs = [1/(1+math.exp(-x)) for x in logits[0]]
print({l: round(p,4) for l,p in zip(LABELS, probs) if p >= thrs[l]})  # fired flags
```

### ARTH quickstart (live Space API)

```python
from gradio_client import Client
c = Client("eulogik/pico-type")
out = c.predict("aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "arth", api_name="/handle_classify")
print(out[7])  # Risk++ (14) tab: probs + [FLAG] markers
```

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
| tiny | 16 | 1.43M | 9.09 MB | 0.65× |
| small | 64 | 1.45M | 9.13 MB | 0.82× |
| base | 192 | 1.48M | 9.25 MB | 1.0× (reference) |
| pro | 576 | 1.56M | 9.61 MB | 1.05× |

ONNX sizes are single-file FP32 exports (graph-only files are 203–206 KB).

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
| **HuggingFace Space** | [eulogik/pico-type](https://huggingface.co/spaces/eulogik/pico-type) | Gradio web UI, no GPU needed — ARTH V2 + Risk++ (14) by default |
| **HuggingFace Model** | [eulogik/pico-type](https://huggingface.co/eulogik/pico-type) | ONNX models + ARTH V2 artifacts (`arth_full_base.onnx`, thresholds, temps, ckpt) |
| **GitHub** | [eulogik/pico-type](https://github.com/eulogik/pico-type) | Source code, training, paper |
| **PyPI** | `pip install picotype` | Python package |
| **ONNX Runtime** | Use with onnxruntime.js | Browser/Node.js deployment |

## 📚 Resources

- [Paper (arXiv)](https://arxiv.org/abs/2608.14658) — Architecture, training, and evaluation details
- [Paper PDF](paper/main.pdf) — Local copy of the paper
- [Model Card](MODEL_CARD.md) — Detailed architecture and training configuration
- [Walkthrough](walkthrough.md) — Development log and decisions
- [Architecture Plan](docs/PLAN.md) — Original design document

## 📄 License

Apache 2.0

---

<div align="center">
<sub>Built with PyTorch · ONNX · Gradio · HuggingFace</sub>
</div>
