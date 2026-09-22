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
- tiny-model
- edge
pipeline_tag: text-classification
library_name: pico-type
arxiv: 2608.14658
metrics:
- accuracy
---

<div align="center">

# pico-type 🔍 (v02)

**A tiny byte-level multi-head content classifier** — ~1.5M params, ~9MB single-file ONNX (FP32), ~18ms CPU inference.

Classifies any content from raw bytes: **coarse type · modality · subtype · code language · text language · file MIME · risk flags**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![PyPI](https://img.shields.io/pypi/v/pico-type?color=blue)](https://pypi.org/project/pico-type/)
[![ONNX](https://img.shields.io/badge/ONNX-exported-success)]()
[![Paper](https://img.shields.io/badge/arXiv-2608.14658-red)](https://arxiv.org/abs/2608.14658)

</div>

---

**Paper**: [pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content Classifier](https://arxiv.org/abs/2608.14658) (arXiv:2608.14658, 2026)

## What's here

Single-file IR-8 ONNX exports (FP32, opset 18) for all 4 Matryoshka tiers — tiny / small / base / pro — sharing the same trunk:

| Tier | Dim | Size | Params |
|------|-----|------|--------|
| tiny | 16 | 9.09 MB | 1,434,344 |
| small | 64 | 9.13 MB | 1,445,480 |
| base | 192 | 9.25 MB | 1,475,176 |
| pro | 576 | 9.61 MB | 1,564,264 |

Load directly with onnxruntime:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("picotype_base.onnx")

text = b"def hello(): pass"
ids = np.frombuffer(text[:1024], dtype=np.uint8).astype(np.int64)
padded = np.zeros(1024, dtype=np.int64)
padded[:len(ids)] = ids
mask = np.zeros(1024, dtype=np.bool_)
mask[:len(ids)] = True

outs = session.run(None, {"input_ids": padded[None, :], "attention_mask": mask[None, :]})
```

## Evaluation (v2 — trained on real data)

| Head | Classes | Accuracy | Dataset |
|------|---------|----------|---------|
| coarse | 12 | **100%** | Synthetic eval |
| modality | 8 | **100%** | Synthetic eval |
| subtype | 24 | **93.8%** | Synthetic eval |
| **code_lang** | **62** | **60.3%** | [The Heap](https://huggingface.co/datasets/AISE-TUDelft/the-heap) — 24 real-world langs |
| **text_lang** | **30** | **98.3%** | [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) — 30 langs |
| file_mime | 90 | **100%** | Synthetic eval |
| risk (multi-label) | 6 | **100%** | Synthetic eval |

## Links

- [arXiv paper](https://arxiv.org/abs/2608.14658)
- [GitHub](https://github.com/eulogik/pico-type)
- [Main HF model repo](https://huggingface.co/eulogik/pico-type)
- [HF Space (live demo)](https://huggingface.co/spaces/eulogik/pico-type)
- PyPI: `pip install pico-type`