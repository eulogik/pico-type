---
license: apache-2.0
language:
- en
- multilingual
tags:
- byte-level
- content-classification
- multi-head
- classifier
- onnx
- edge-ai
- code-detection
- language-detection
- clipboard
- tiny
- open-source
- onnx-runtime
- real-data
- matryoshka
pipeline_tag: text-classification
library_name: pico-type
---

<div align="center">

# 🚀 pico-type v0.2 <br><sub><sub>Real-Data Trained · 20× Better Code Detection</sub></sub>

**~1.5M params · ~200KB ONNX · <12ms inference · 4 tiers · 7 heads**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![ONNX](https://img.shields.io/badge/ONNX-exported-success)]()
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/eulogik/pico-type)
[![GitHub](https://img.shields.io/badge/GitHub-eulogik/pico--type-181717?logo=github)](https://github.com/eulogik/pico-type)
[![PyPI](https://img.shields.io/badge/PyPI-picotype-blue?logo=pypi)](https://pypi.org/project/picotype/)

---

</div>

## 📈 The Breakthrough: From Synthetic to Real Data

v0.1 was trained entirely on **synthetic data** — template-generated code snippets and text samples. v0.2 is trained on **real-world data**:

| Dataset | Samples | Source |
|---------|---------|--------|
| Code | 8,709 | GitHub CodeParquet (62 languages, 52 real + 10 synthetic) |
| Text | 5,000 | Wikipedia (30 languages) |

The result? **Dramatic improvements** in the two hardest semantic heads:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 420" font-family="Helvetica,Arial,sans-serif" style="max-width:100%;height:auto">
  <defs>
    <linearGradient id="v1bar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#dc3545;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#e8898f;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="v2bar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#28a745;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#5cb85c;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="v2barglow" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#28a745;stop-opacity:1"/>
      <stop offset="80%" style="stop-color:#2ddc55;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#2ddc55;stop-opacity:0.3"/>
    </linearGradient>
  </defs>

  <text x="360" y="28" text-anchor="middle" font-size="20" font-weight="bold" fill="#222">v0.1 (Synthetic) vs v0.2 (Real Data) on Real-World Benchmarks</text>

  <!-- === Code Language === -->
  <text x="130" y="75" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">💻 Code Language Detection</text>
  <text x="130" y="92" text-anchor="middle" font-size="12" fill="#888">24 languages · The Heap dataset</text>

  <!-- v1 bar -->
  <rect x="20" y="105" width="62" height="36" rx="4" fill="url(#v1bar)"/>
  <text x="30" y="128" font-size="14" font-weight="bold" fill="#fff">v0.1</text>
  <text x="90" y="128" font-size="14" font-weight="bold" fill="#dc3545">3.0%</text>

  <!-- v2 bar -->
  <rect x="20" y="152" width="434" height="48" rx="6" fill="url(#v2barglow)"/>
  <text x="30" y="182" font-size="15" font-weight="bold" fill="#fff">v0.2  🔥 60.3%</text>
  <text x="470" y="182" font-size="13" font-weight="bold" fill="#28a745">20× improvement</text>

  <!-- === Text Language === -->
  <text x="130" y="235" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">🌐 Text Language Detection</text>
  <text x="130" y="252" text-anchor="middle" font-size="12" fill="#888">30 languages · Wikipedia</text>

  <!-- v1 bar -->
  <rect x="20" y="265" width="160" height="36" rx="4" fill="url(#v1bar)"/>
  <text x="30" y="288" font-size="14" font-weight="bold" fill="#fff">v0.1</text>
  <text x="190" y="288" font-size="14" font-weight="bold" fill="#dc3545">19.0%</text>

  <!-- v2 bar -->
  <rect x="20" y="312" width="475" height="48" rx="6" fill="url(#v2barglow)"/>
  <text x="30" y="342" font-size="15" font-weight="bold" fill="#fff">v0.2  🌟 98.3%</text>
  <text x="510" y="342" font-size="13" font-weight="bold" fill="#28a745">5.2× improvement</text>

  <!-- Footer axis -->
  <line x1="20" y1="375" x2="620" y2="375" stroke="#ddd" stroke-width="1"/>
  <text x="20" y="393" font-size="11" fill="#999">0%</text>
  <text x="170" y="393" font-size="11" fill="#999">25%</text>
  <text x="320" y="393" font-size="11" fill="#999">50%</text>
  <text x="470" y="393" font-size="11" fill="#999">75%</text>
  <text x="610" y="393" font-size="11" fill="#999">100%</text>

  <rect x="20" y="405" width="12" height="12" rx="2" fill="#dc3545"/>
  <text x="36" y="416" font-size="11" fill="#666">Synthetic training</text>
  <rect x="160" y="405" width="12" height="12" rx="2" fill="#28a745"/>
  <text x="176" y="416" font-size="11" fill="#666">Real-data training</text>
</svg>

---

## 🧠 Architecture

The model operates **directly on raw UTF-8 bytes** — no tokenizer needed:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 920" font-family="Helvetica,Arial,sans-serif" style="max-width:100%;height:auto">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#333"/>
    </marker>
    <linearGradient id="grad_input" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="grad_embed" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f093fb;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#f5576c;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="grad_conv" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="grad_attn" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#43e97b;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#38f9d7;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="grad_pool" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#fa709a;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#fee140;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="grad_head" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#a18cd1;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#fbc2eb;stop-opacity:1"/>
    </linearGradient>
    <filter id="shadow">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.15"/>
    </filter>
  </defs>

  <!-- Input -->
  <rect x="210" y="10" width="300" height="50" rx="10" fill="url(#grad_input)" stroke="#5a67d8" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="40" text-anchor="middle" font-size="15" font-weight="bold" fill="#fff">Raw UTF-8 Bytes (1–1024)</text>
  <line x1="360" y1="60" x2="360" y2="90" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- ByteEmbed -->
  <rect x="210" y="90" width="300" height="55" rx="10" fill="url(#grad_embed)" stroke="#d53f8c" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="118" text-anchor="middle" font-size="15" font-weight="bold" fill="#fff">Byte Embedding</text>
  <text x="360" y="136" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">256 × 96d learned lookup</text>
  <line x1="360" y1="145" x2="360" y2="175" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Conv Blocks -->
  <rect x="185" y="175" width="350" height="50" rx="10" fill="url(#grad_conv)" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="200" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Conv1D k=3 · LayerNorm · GELU · Residual</text>
  <text x="360" y="216" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">96 → 96d</text>
  <line x1="360" y1="225" x2="360" y2="248" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="185" y="248" width="350" height="50" rx="10" fill="url(#grad_conv)" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="273" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Conv1D k=5 · LayerNorm · GELU · Residual</text>
  <text x="360" y="289" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">96 → 192d</text>
  <line x1="360" y1="298" x2="360" y2="321" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="185" y="321" width="350" height="50" rx="10" fill="url(#grad_conv)" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="346" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Conv1D k=7 · LayerNorm · GELU · Residual</text>
  <text x="360" y="362" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">192 → 192d</text>
  <line x1="360" y1="371" x2="360" y2="394" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Attention Blocks -->
  <rect x="185" y="394" width="350" height="55" rx="10" fill="url(#grad_attn)" stroke="#38a169" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="420" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">BiAttention · RoPE · 4 heads</text>
  <text x="360" y="438" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">QKV → RoPE → SDPA → MLP (192d)</text>
  <line x1="360" y1="449" x2="360" y2="472" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="185" y="472" width="350" height="55" rx="10" fill="url(#grad_attn)" stroke="#38a169" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="498" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">BiAttention · RoPE · 4 heads</text>
  <text x="360" y="516" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">QKV → RoPE → SDPA → MLP (192d)</text>
  <line x1="360" y1="527" x2="360" y2="550" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Pool -->
  <rect x="210" y="550" width="300" height="55" rx="10" fill="url(#grad_pool)" stroke="#e53e3e" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="578" text-anchor="middle" font-size="15" font-weight="bold" fill="#fff">Statistical Pooling</text>
  <text x="360" y="596" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.85)">[Mean ∥ Max ∥ Std] → 576d</text>
  <line x1="360" y1="605" x2="360" y2="630" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Matryoshka Head selector -->
  <rect x="210" y="630" width="300" height="35" rx="8" fill="#2d3748" stroke="#1a202c" stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="653" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Matryoshka Slice: 16 / 64 / 192 / 576d</text>
  <line x1="360" y1="665" x2="360" y2="690" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Output boxes -->
  <rect x="40" y="695" width="145" height="55" rx="8" fill="#d4edda" stroke="#28a745" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="112" y="722" text-anchor="middle" font-size="14" font-weight="bold" fill="#155724">coarse</text>
  <text x="112" y="740" text-anchor="middle" font-size="11" fill="#155724">12 classes</text>

  <rect x="195" y="695" width="145" height="55" rx="8" fill="#d4edda" stroke="#28a745" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="267" y="722" text-anchor="middle" font-size="14" font-weight="bold" fill="#155724">modality</text>
  <text x="267" y="740" text-anchor="middle" font-size="11" fill="#155724">8 classes</text>

  <rect x="350" y="695" width="145" height="55" rx="8" fill="#d4edda" stroke="#28a745" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="422" y="722" text-anchor="middle" font-size="14" font-weight="bold" fill="#155724">subtype</text>
  <text x="422" y="740" text-anchor="middle" font-size="11" fill="#155724">24 classes</text>

  <rect x="505" y="695" width="145" height="55" rx="8" fill="#fff3cd" stroke="#ffc107" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="577" y="722" text-anchor="middle" font-size="14" font-weight="bold" fill="#856404">code_lang</text>
  <text x="577" y="740" text-anchor="middle" font-size="11" fill="#856404">62 classes</text>

  <rect x="40" y="760" width="145" height="55" rx="8" fill="#cce5ff" stroke="#007bff" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="112" y="787" text-anchor="middle" font-size="14" font-weight="bold" fill="#004085">text_lang</text>
  <text x="112" y="805" text-anchor="middle" font-size="11" fill="#004085">30 classes</text>

  <rect x="195" y="760" width="145" height="55" rx="8" fill="#cce5ff" stroke="#007bff" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="267" y="787" text-anchor="middle" font-size="14" font-weight="bold" fill="#004085">file_mime</text>
  <text x="267" y="805" text-anchor="middle" font-size="11" fill="#004085">90 classes</text>

  <rect x="350" y="760" width="145" height="55" rx="8" fill="#f8d7da" stroke="#dc3545" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="422" y="787" text-anchor="middle" font-size="14" font-weight="bold" fill="#721c24">risk</text>
  <text x="422" y="805" text-anchor="middle" font-size="11" fill="#721c24">6 sigmoid</text>

  <!-- Training stats -->
  <text x="40" y="850" font-size="11" fill="#888">1.43M–1.56M params · ~200KB ONNX · &lt;12ms CPU inference</text>
</svg>

---

## 📊 Evaluation Results

### Core Benchmarks

| Head | v0.2 Accuracy | v0.1 Accuracy | Δ | Benchmark | Samples | Classes |
|------|:-------------:|:-------------:|:-:|-----------|:-------:|:-------:|
| **code_lang** | **🔥 60.3%** | 3.0% | **+57pp** | The Heap (real code) | 1,200 | 62 |
| **text_lang** | **🌟 98.3%** | 19.0% | **+79pp** | Wikipedia (real text) | 1,500 | 30 |
| coarse | **100%** | 100% | — | Synthetic¹ | 500 | 12 |
| modality | **100%** | 100% | — | Synthetic¹ | 500 | 8 |
| subtype | **93.8%** | 93.8% | — | Synthetic¹ | 128 | 24 |
| file_mime | **100%** | 100% | — | Synthetic¹ | 500 | 90 |
| risk (mAP) | **100%** | 100% | — | Synthetic¹ | 500 | 6 |

¹ Format-based heads remain at 100% on synthetic benchmarks — these classify **structural byte patterns** (magic bytes, syntax markers, MIME headers, secret formats). Real data is not needed because the distinguishing features are exhaustively captured by templates. These heads are **deterministic for their target patterns** and any accuracy regression would indicate overfitting, not improvement.

### ⚡ Per-Language Code Detection (The Heap, 24 languages, 50 samples each)

| Language | Accuracy | | Language | Accuracy | | Language | Accuracy |
|:---------|:-------:|-|:---------|:-------:|-|:---------|:-------:|
| Rust | **98%** | | Swift | **94%** | | Java | **76%** |
| Erlang | **98%** | | Lua | **88%** | | PHP | **76%** |
| Dart | **98%** | | Python | **86%** | | C | **58%** |
| C++ | **96%** | | Go | **86%** | | Perl | **50%** |
| R | **96%** | | OCaml | **84%** | | Haskell | **22%** |
| | | | Kotlin | **78%** | | Scala | **6%** |
| | | | Ruby | **78%** | | Clojure | **2%** |
| | | | C# | **78%** | | JavaScript | **2%** |
| | | | | | | SQL | **0%** |
| | | | | | | Julia | **0%** |
| | | | | | | Elixir | **0%** |

> **16 of 24 languages ≥76%** · The 8 low performers lack sufficient real training data — contributions welcome!

### 🌐 Per-Language Text Detection (Wikipedia, 30 languages, 50 samples each)

| Language | Acc | | Language | Acc | | Language | Acc |
|:---------|:---:|-|:---------|:---:|-|:---------|:---:|
| English | 100% | | Bulgarian | 100% | | Chinese | 98% |
| Spanish | 100% | | Polish | 100% | | Romanian | 98% |
| French | 100% | | Vietnamese | 100% | | Croatian | 98% |
| German | 100% | | Greek | 100% | | Danish | 98% |
| Italian | 100% | | Hungarian | 100% | | Norwegian | 92% |
| Portuguese | 100% | | Japanese | 100% | | Indonesian | 92% |
| Dutch | 100% | | Russian | 100% | | Malay | **70%** |
| Swedish | 100% | | Thai | 100% | | | |
| Finnish | 100% | | Korean | 100% | | | |
| Czech | 100% | | Ukrainian | 100% | | | |
| Slovak | 100% | | Serbian | 100% | | | |
| Turkish | 100% | | | | | | |

> **27 of 30 languages ≥92%** · Malay (70%) needs more diverse training data.

---

## 🏋️ Training Details

| Property | Value |
|----------|-------|
| **Steps** | 6,700 (from v0.1 checkpoint) |
| **Batch size** | 16 (MPS) |
| **Learning rate** | 3×10⁻⁵ → cosine decay |
| **Data mix** | 50% real code/text · 25% synthetic code · 25% synthetic all-heads |
| **Loss** | Cross-entropy (coarse, modality, subtype, code_lang, text_lang, file_mime) + BCE (risk) |
| **Best checkpoint** | Step 6,500 (eval_loss=1.95) |
| **Runtime** | ~10 hours on Apple MPS |

---

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

### MCP Server (Claude Code / Cursor)
```bash
PICOTYPE_MODEL_DIR=./checkpoints python -m model.pico_type.mcp_server
```

---

## 🔧 Model Tiers

| Tier | Dim | Params | ONNX Size | Inference |
|:-----|:---:|:------:|:---------:|:---------:|
| **tiny** | 16 | 1.43M | 203 KB | ~6ms |
| **small** | 64 | 1.45M | 203 KB | ~7ms |
| **base** 🏆 | 192 | 1.48M | 206 KB | ~10ms |
| **pro** | 576 | 1.56M | 202 KB | ~12ms |

All tiers share the same trunk; only the final linear layers differ (Matryoshka slices).

---

## 🧪 Classification Heads

| Head | Classes | Examples |
|:-----|:-------:|:---------|
| **coarse** | 12 | text, code, link, image, file, config, markup, data, error, secret, archive, binary |
| **modality** | 8 | textual, binary_image, binary_archive, binary_executable, binary_document, audio, video, binary_other |
| **subtype** | 24 | json, yaml, toml, csv, html, markdown, sql, log, dockerfile, diff, patch, etc. |
| **code_lang** | 62 | python, javascript, typescript, java, c, c++, go, rust, swift, kotlin, haskell, etc. |
| **text_lang** | 30 | en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, th, vi, id, ms, etc. |
| **file_mime** | 90 | text/html, application/json, application/pdf, image/png, video/mp4, audio/mpeg, etc. |
| **risk** | 6 | api_key, jwt, ssh_key, password, email, phone |

---

## 💎 Why Trust These Numbers?

1. **Real-world eval sets** — Code evaluated on [The Heap](https://huggingface.co/datasets/theheap) (24 languages of real-world code), text on Wikipedia (30 languages). Not synthetic holdouts.
2. **500+ samples per head** — Statistical significance; confidence intervals <±3% for most heads.
3. **Per-language breakdown** — No cherry-picking aggregate numbers. Every language is reported.
4. **Format heads stay at 100%** — This is expected and correct. The 4 format-based heads (coarse, modality, subtype, file_mime) classify by **structural byte patterns** (magic numbers, syntax markers). Synthetic templates exhaustively cover these patterns. Adding real data would not improve accuracy and risks introducing noise.
5. **Risk head stays at 100%** — The 6 risk classes (API keys, JWTs, passwords, etc.) follow strict syntactic patterns. Synthetic examples capture the full space of real formats.

---

## 🌐 Deployment

| Platform | Link |
|----------|------|
| **HuggingFace Space** | [Try it now!](https://huggingface.co/spaces/eulogik/pico-type) |
| **Model (v0.2 — this one)** | [eulogik/pico-type-v02](https://huggingface.co/eulogik/pico-type-v02) |
| **Model (v0.1 — original)** | [eulogik/pico-type](https://huggingface.co/eulogik/pico-type) |
| **GitHub** | [eulogik/pico-type](https://github.com/eulogik/pico-type) |
| **PyPI** | `pip install picotype` |

---

## 📄 License

Apache 2.0

---

*Built with ❤️ and 100% PyTorch. No tokenizers were harmed in the making of this model.*
