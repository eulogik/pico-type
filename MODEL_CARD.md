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
- content-moderation
- risk-detection
- secrets-detection
- prompt-injection-detection
- pii-detection
pipeline_tag: text-classification
library_name: pico-type
arxiv: 2608.14658
model-index:
- name: pico-type ARTH V2 (composite)
  results:
  - task:
      type: text-classification
      name: Risk flagging, 14 labels
    dataset:
      name: ARTH external audit (held-out, n=211, never trained or fit on)
      type: private
    metrics:
    - type: recall
      name: positive-class recall
      value: 0.536
    - type: specificity
      name: specificity
      value: 0.944
    - type: balanced error
      name: balanced error
      value: 0.260
  - task:
      type: text-classification
      name: ToxicChat jailbreak gate (held-out test split)
    dataset:
      name: ToxicChat test (toxicchat0124)
      type: lmsys/toxic-chat
    metrics:
    - type: recall
      name: recall@threshold
      value: 0.824
    - type: accuracy
      name: selective accuracy at 50% coverage
      value: 0.973
---

# pico-type

A tiny **byte-level multi-head content classifier** (~1.5M parameters) that classifies any content into **7 categories** simultaneously from raw bytes — no tokenizer, no pretrained embeddings.

> **Paper**: [pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content Classifier](https://arxiv.org/abs/2608.14658) (arXiv:2608.14658, 2026)

## Architecture

```
ByteEmbed → Conv1D×3 → BiAttention×2 → Pool → Matryoshka Heads
```

- **Byte-level**: operates directly on UTF-8 bytes, supports any language
- **Matryoshka heads**: 7 independent classification heads with 4 tiers (tiny/small/base/pro)
- **1.5M params**: fits in ~9MB single-file ONNX (FP32), runs in ~18ms on CPU
- **No tokenizer**: zero vocabulary dependencies

## Classification Heads

| Head | Classes | Description |
|------|---------|-------------|
| **coarse** | 12 | text, code, link, image, file, config, markup, data, error, secret, archive, binary |
| **modality** | 8 | textual, binary_image, binary_archive, binary_executable, binary_document, binary_audio, binary_video, binary_other |
| **subtype** | 24 | json, yaml, toml, ini, csv, html, xml, markdown, sql, log, diff, dockerfile, etc. |
| **code_lang** | 62 | python, javascript, typescript, java, c, cpp, go, rust, swift, bash, sql, etc. |
| **text_lang** | 30 | en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, etc. |
| **file_mime** | 90 | text/html, application/json, application/pdf, image/png, video/mp4, etc. |
| **risk** | 6 | api_key, jwt, password, email, phone, ssh_key (probabilities) |

## Performance

Benchmarked on synthetic data (500 samples, 1024 bytes max, base tier, 1700 training steps):

| Head | Accuracy | Support |
|------|----------|---------|
| coarse | 100.0% | 500 |
| modality | 100.0% | 500 |
| subtype | 93.8% | 128 |
| code_lang | 41.7% | 48 |
| text_lang | 94.3% | 35 |
| file_mime | 100.0% | 131 |
| risk (mAP) | 100.0% | — |

- **Inference**: ~18ms per sample on CPU (ONNX Runtime, M2)
- **Model size**: ~9.25MB single-file FP32 (base tier; 206 KB graph + 9.05 MB weights)
- **Loss**: 1.97 eval_loss (best, step 1700)

> **code_lang** accuracy (54.2%) reflects 62-class coverage; improves with longer sequences (>256 bytes). v0.2 will target better code language discrimination.

## ARTH V2 — Risk++ (current release)

Frozen-trunk composite: ft3 student + retrained 14-label **Risk++** head family (api_key, jwt, ssh_key, password, email, phone, prompt_injection, jailbreak, pii_ssn, pii_card, secrets_aws, secrets_github, sql_injection, xss_payload) with per-label fitted thresholds. **No trunk or semantic/relational drift** — the composite differs from the base model only under `riskpp.*`, so every gate below reproduces the base model byte-for-byte except risk.

| Metric | Before | ARTH V2 |
|---|---|---|
| External-audit positive recall (held-out, never fit on) | 0.107 (12/112) | **0.536** (60/112) |
| Balanced error | 0.459 | **0.260** |
| Specificity | 0.961 | 0.944 |
| Fitted label thresholds | 9 | **14** |
| Legacy parity / choice / noul | 60/60 / 0.737 / 0.929 | unchanged |
| ToxicChat recall / sel-acc@50 | 0.846 / 0.970 | **0.824 / 0.973** (gate PASS) |

Artifacts ([huggingface.co/eulogik/pico-type](https://huggingface.co/eulogik/pico-type)): `arth_full_base.onnx` (**11.66 MB**, opset 18, IR 8; auto-verify PASS — torch-vs-ORT err 1.1e-05, parity 60/60, semantic 300/300, risk flags 11/11), `arth_full_base_int8.onnx` (4.28 MB, experimental), `risk_thresholds.json`, `temperatures.json`, `arth_final_composite.pt`. Live demo: [Space](https://huggingface.co/spaces/eulogik/pico-type) (`arth` model + Risk++ (14) tab). See README for the copy-paste ONNX + Space-API quickstarts.

**Known limitations (measured 2026-09):** pii_ssn 0.125, api_key 0.625, jailbreak 0.50 on embedded phrasing; 3/11 hand-picked benign probes still flag (incl. accepted `print('hello world')`→sql); long-document signals dilute in fixed-window pooling; general-classification gates (AG/SST-2/Enron) stay at chance — a byte-pattern risk flagger, not a general text classifier.

## Usage

### CLI
```bash
# Pipe content
echo "def hello(): pass" | picotype --pretty

# File
picotype --file document.txt

# Clipboard (macOS)
picotype --clip
```

### Python
```python
from model.pico_type.labels import decode_output

# Run with ONNX session
result = {"coarse": "code", "modality": "textual", ...}
decoded = decode_output(result, tier="base")
```

### MCP Server
```bash
PICOTYPE_MODEL_DIR=./checkpoints python -m model.pico_type.mcp_server
```

## Model Tiers

| Tier | Head Dim | Params | ONNX Size |
|------|----------|--------|-----------|
| tiny | 16 | 1.43M | 9.09 MB |
| small | 64 | 1.45M | 9.13 MB |
| base | 192 | 1.48M | 9.25 MB |
| pro | 576 | 1.56M | 9.61 MB |

All tiers share the same trunk; only the final linear layer differs per tier.

## Deployment

### HuggingFace Space
The [Gradio Space](https://huggingface.co/spaces/eulogik/pico-type) provides:
- Text input and file upload
- Real-time 7-head classification
- Tier selection (tiny/small/base/pro)
- **ARTH V2 (default)**: same 7 heads plus a Risk++ (14) tab with fitted thresholds and `[FLAG]` markers

### ONNX Runtime
```python
import onnxruntime
session = ort.InferenceSession("picotype_base.onnx")
```

## Training

Trained on synthetic data (11 content buckets, 62 code languages, 30 text languages, 90 MIME types) using multi-task loss with 500 optimization steps.

- **Loss**: weighted cross-entropy (coarse) + binary cross-entropy (risk)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **GPU**: ~100ms/step on MPS, ~3.5s/step on CPU

## License

Apache 2.0
