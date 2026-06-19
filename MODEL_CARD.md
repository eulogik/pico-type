---
license: apache-2.0
language:
- multilingual
tags:
- content-classification
- byte-level
- onnx
- matryoshka
- lightweight
- classifier
- multi-head
- clipboard
pipeline_tag: text-classification
library_name: pico-type
co2_eq_emissions: 0.02
model-index:
- name: pico-type
  results:
  - task:
      type: text-classification
      name: Multi-Head Content Classification
    metrics:
      - type: accuracy
        value: 0.998
        name: Coarse Accuracy
      - type: accuracy
        value: 1.0
        name: Modality Accuracy
      - type: accuracy
        value: 0.947
        name: Subtype Accuracy
      - type: accuracy
        value: 0.387
        name: Code Language Accuracy
      - type: accuracy
        value: 0.988
        name: Text Language Accuracy
      - type: accuracy
        value: 1.0
        name: File MIME Accuracy
      - type: average_precision
        value: 1.0
        name: Risk Average Precision
---

# pico-type

**A tiny byte-level multi-head content classifier** (~1.5M parameters) built by [**eulogik**](https://eulogik.com).

Classifies any content into **7 categories** simultaneously from raw bytes in a single forward pass — no tokenizer, no pretrained embeddings, no GPU required.

## Model Details

| Property | Value |
|----------|-------|
| **Developer** | [eulogik](https://eulogik.com) |
| **Release Date** | June 2026 |
| **License** | Apache 2.0 |
| **Model Type** | Byte-level neural classifier |
| **Input** | Raw UTF-8 bytes (0–255), up to 1024 bytes |
| **Output** | 7 classification heads |
| **Parameters** | 1.43M–1.56M (4 Matryoshka tiers) |
| **ONNX Size** | ~200 KB (FP32) |
| **Inference** | ~5ms on CPU (base tier) |
| **Training Hardware** | Apple M4 (MPS), 16GB |

## Architecture

```
ByteEmbed(256→96d) → 3×Conv1D(k=3,5,7) → 2×BiAttention(RoPE) → Pool → 7×Matryoshka Heads
```

- **Byte-level**: No tokenizer, no subword vocabulary, no pretrained embeddings
- **Matryoshka**: 4 tiered variants share the same trunk; head dimensions vary (16/64/192/576)
- **Multi-head**: 7 independent classification heads trained jointly with per-task loss weighting

## Classification Heads

| Head | Classes | Gated | Examples |
|------|---------|-------|----------|
| **coarse** | 12 | — | text, code, link, image, file, config, markup, data, error, secret, archive, binary |
| **modality** | 8 | — | textual, binary_image, binary_archive, binary_executable, binary_document, binary_audio, binary_video, binary_other |
| **subtype** | 24 | coarse∈{config,markup,data} | json, yaml, toml, ini, csv, html, xml, markdown, sql, log, diff, dockerfile |
| **code_lang** | 62 | coarse=code | python, javascript, typescript, java, c, cpp, go, rust, swift, kotlin, bash, sql, ruby, php, perl, lua, r, julia, haskell, scala, dart, elixir, clojure, erlang, zig, nim |
| **text_lang** | 30 | coarse=text | en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, tr, nl, pl, sv, da, fi, nb, cs, hu, ro, uk, el, he, th, vi, id, ms, ta |
| **file_mime** | 90 | coarse∈{image,file} | text/html, application/json, application/pdf, image/png, image/jpeg, video/mp4, audio/mpeg, application/zip |
| **risk** | 6 | — | api_key, jwt, password, email, phone, ssh_key (independent probabilities) |

## Performance

### Synthetic Benchmark (1000 samples, base tier)

| Head | Accuracy | Support |
|------|----------|---------|
| coarse | 99.8% | 1000 |
| modality | 100.0% | 1000 |
| subtype | 94.7% | ~250 |
| code_lang | 38.7% | ~90 |
| text_lang | 98.8% | ~80 |
| file_mime | 100.0% | ~250 |
| risk (mAP) | 100.0% | — |

**Inference**: 5.0ms on M4 CPU via ONNX Runtime (base tier, 1024 bytes)

### Real-World Evaluation (21 hand-curated inputs)

| Category | Accuracy |
|----------|----------|
| Coarse | **86%** (18/21) |
| Modality | **100%** (21/21) |
| Overall | **52%** (11/21 exact match) |

_Diverse training improved real-world accuracy from 23% to 52%._

### Tier Comparison

| Tier | Dim | Params | ONNX Size | Speed |
|------|-----|--------|-----------|-------|
| tiny | 16 | 1.43M | 207 KB | ~3ms |
| small | 64 | 1.45M | 207 KB | ~4ms |
| base | 192 | 1.48M | 209 KB | ~5ms |
| pro | 576 | 1.56M | 206 KB | ~12ms |

## Training

The model is trained using a multi-task loss:

```
L = Σ w_h · L_h
```

- **Single-label heads**: Cross-entropy with ignore_index=-100 for gated samples
- **Risk head**: Binary cross-entropy
- **Head weights**: coarse=8.0, modality=2.0, code_lang=2.0, text_lang=1.5, others=1.0

### Training Data

1. **Synthetic data** — 11 content buckets with per-language templates (62 code langs, 30 text langs, 90 MIME types, 22 binary formats)
2. **Diverse synthetic** — Realistic error traces, markdown with code blocks, nested configs, environment files, multi-language prose
3. **Real GitHub code** — ~650 files across 62 languages via GitHub Search API (30% training mix)

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 3e-3 (cosine decay) |
| Warmup Steps | 200 |
| Weight Decay | 0.01 (trunk), 0.0 (heads) |
| Batch Size | 16 |
| Total Steps | 4000 |
| Gradient Clip | 1.0 |
| Hardware | Apple M4 16GB (MPS) |

## Usage

### CLI
```bash
pip install picotype
echo "def hello(): pass" | picotype --pretty
picotype --file document.txt
picotype --clip
```

### Python
```python
from picotype import PicoType, PicoTypeConfig, load_checkpoint, decode_output

model = PicoType(PicoTypeConfig()).eval()
load_checkpoint("checkpoints/best.pt", model)
logits = model(encode_bytes(b"input content"))
result = decode_output(logits)
```

### ONNX Runtime
```python
import onnxruntime as ort
session = ort.InferenceSession("picotype_base.onnx")
# input: int64[1, seq_len], bool[1, seq_len]
```

### MCP Server
```bash
PICOTYPE_MODEL_DIR=./checkpoints python -m model.pico_type.mcp_server
```

## Model Tiers

All tiers share the same trunk; only the Matryoshka head linears differ. You can switch tiers at inference time with zero overhead:

```python
logits_tiny = model(x, mask, tier="tiny")
logits_base = model(x, mask, tier="base")
```

## Deployment

| Platform | URL |
|----------|-----|
| HuggingFace Space | [eulogik/pico-type](https://huggingface.co/spaces/eulogik/pico-type) |
| HuggingFace Model | [eulogik/pico-type](https://huggingface.co/eulogik/pico-type) |
| GitHub | [eulogik/pico-type](https://github.com/eulogik/pico-type) |
| PyPI | `pip install picotype` |
| Zenodo | [10.5281/zenodo.20758542](https://doi.org/10.5281/zenodo.20758542) |

## Limitations

- **code_lang accuracy**: 62-way classification with minimal per-class support. Real GitHub data helps (+10% absolute) but authentic code patterns (imports, type hints, decorators) remain challenging.
- **Synthetic-only training**: Model overfits to template patterns. Diverse synthetic + real data significantly improves real-world robustness.
- **Max input length**: 1024 bytes. Longer content is truncated.
- **No fine-grained text understanding**: Designed for content classification, not NLP tasks like sentiment analysis or NER.

## Citation

```bibtex
@software{eulogik2026picotype,
  author = {eulogik},
  title = {pico-type: A Tiny Byte-Level Multi-Head Content Classifier},
  year = {2026},
  url = {https://github.com/eulogik/pico-type},
  doi = {10.5281/zenodo.20758542}
}
```

---

<div align="center">
  <sub>Built by <a href="https://eulogik.com">eulogik</a> — AI infrastructure for developers.</sub>
</div>
