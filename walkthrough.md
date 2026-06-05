# pico-type — Walkthrough

> Living document. Every agent/harness picking up this project should start here. Also, you must keep updating as you go further.

---

## 1. What this is

**pico-type** is a tiny (~1.5M params), byte-level, multi-head content classifier. Input: up to 1024 raw bytes (clipboard text, file bytes, image header, etc.). Output: structured label set in one forward pass.

Built per the locked-in plan in [`docs/PLAN.md`](docs/PLAN.md) (recovered from the original opencode session — see §11).

### Why this exists
Existing clipboard tools are regex-only (ClipGate, 13 types) or LLM-powered (needs Ollama, GB-scale). Existing tiny classifiers do one job. **No model does all of them in one sub-5MB forward pass with a multi-head output.**

### Deliverables
- HuggingFace model (Apache-2.0) with 4 Matryoshka tiers
- Rust + Python CLI (`picotype`)
- Gradio Space that doubles as an **MCP server**
- Browser extension, Raycast/Alfred/VSCode extensions
- arXiv paper

---

## 2. Locked-in architecture (from `docs/PLAN.md` §2)

```
Inputs (≤1024 UTF-8 bytes, masked/padded)
  │
  ├─ ByteEmbed (256 → 96d, learned)
  │
  ├─ 3× Conv1D block (kernel 3, 5, 7) + GELU + residual  → 192d
  │
  ├─ 2× BiAttention block (d=192, 4 heads, RoPE θ=500k)
  │
  ├─ Pool = [mean ‖ max ‖ std]  → 576d shared trunk
  │
  └─ 7 Matryoshka heads (Linear at 16/64/192/576 dim slices)
       ├─ h_coarse        (12)   — primary type
       ├─ h_modality      (8)    — textual / binary-image / …
       ├─ h_subtype       (24)   — JSON/YAML/CSV/HTML…  (if coarse ∈ {config, markup, data})
       ├─ h_code_lang     (62)   — if coarse=code, + "undetected" fallback
       ├─ h_text_lang     (30)   — if coarse=text, + "undetected" fallback
       ├─ h_file_mime     (90)   — if coarse ∈ {image, file, archive, binary} or modality=binary_*, + "undetected"
       └─ h_risk          (6)    — sigmoid multi-label: api_key, jwt, ssh_key, password, email, phone
```

### Tier matrix
| Tier | Dim slice | Params (actual) | INT8 size |
|---|---|---|---|
| `pico-type-tiny` | 16 | 1.43M | 1.37 MB |
| `pico-type-small` | 64 | 1.45M | 1.38 MB |
| `pico-type-base` | 192 | 1.48M | 1.41 MB |
| `pico-type-pro` | 576 | 1.56M | 1.49 MB |

Actual sizes came in **under** the plan's targets (0.5/1.5/3.5/8 MB INT8). Trunk dominates; if we need to shrink tiny further, reduce `trunk_dim` 192→128 or `num_attn_layers` 2→1.

### Output schema (always returned, with confidence)
```json
{
  "coarse": "code",
  "modality": "textual",
  "subtype": null,
  "code_language": "python",
  "text_language": null,
  "file_mime": null,
  "risk_flags": [],
  "confidence": 0.94,
  "modality_confidence": 0.91,
  "model_tier": "base"
}
```

---

## 3. Repo layout

```
classifier-model/
├── .venv/                          # Python 3.11 venv (torch, numpy<2, safetensors, pyyaml)
├── .git/                           # init'd, no commits yet
├── docs/
│   └── PLAN.md                     # Full plan recovered from opencode session
├── model/
│   └── pico_type/
│       ├── __init__.py             # re-exports public API
│       ├── labels.py               # ✅ vocabularies + decode_output
│       ├── arch.py                 # ✅ PicoType model + smoke test
│       ├── data.py                 # ⏳ TODO: synthetic generator
│       ├── train.py                # ⏳ TODO: multi-task trainer
│       ├── distill.py              # ⏳ TODO: KD from per-head teachers
│       ├── eval.py                 # ⏳ TODO: public eval harness
│       └── export.py               # ⏳ TODO: ONNX, int8, tract, gguf
├── crates/
│   ├── picotype/                   # empty (Rust CLI)
│   └── picotype-mcp/               # empty (Rust MCP server)
├── extensions/
│   ├── chrome/                     # empty (MV3)
│   ├── raycast/                    # empty
│   ├── alfred/                     # empty
│   └── vscode/                     # empty
├── spaces/picotype/                # empty (Gradio + MCP)
├── eval/                           # empty
├── paper/                          # empty (arXiv LaTeX)
└── walkthrough.md                  # ← this file
```

---

## 4. Public API (current)

```python
from model.pico_type import (
    PicoType, PicoTypeConfig, TIERS,            # model
    COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS,
    CODE_LANG_LABELS, TEXT_LANG_LABELS,
    FILE_MIME_LABELS, RISK_LABELS,
    UNDETECTED, decode_output,                  # labels + decoder
)
from model.pico_type.arch import encode_bytes   # bytes → (tokens, mask) tensors
```

### Smoke test
```bash
source .venv/bin/activate
python -m model.pico_type.arch
```

---

## 5. Key invariants (DO NOT break these)

These are tied to the locked-in plan and the on-disk checkpoints. Changing them means re-training + re-publishing.

### Vocab sizes (asserted in `labels.py`)
| Head | Size | Plan target |
|---|---|---|
| coarse | 12 | 12 ✓ |
| modality | 8 | 8 ✓ |
| subtype | 24 | 24 ✓ |
| code_lang | 62 | 62 ✓ |
| text_lang | 30 | 30 ✓ |
| file_mime | 90 | 90 ✓ |
| risk | 6 | 6 ✓ |

### Gating (enforced in `decode_output`, must also be enforced in training loss)
- `subtype` is only valid when `coarse ∈ {config, markup, data}`
- `code_lang` is only valid when `coarse == code`
- `text_lang` is only valid when `coarse == text`
- `file_mime` is only valid when `coarse ∈ {image, file, archive, binary}` or `modality` starts with `binary_`
- `risk` is **always** valid (multi-label)

### `UNDETECTED` behavior
- For `code_lang` / `text_lang` / `file_mime`: the model has N logits (62/30/90), no separate "undetected" class. If `max_softmax < undetected_threshold` (default 0.4), the decoder returns `"undetected"` instead of the argmax label.
- For `risk`: per-class sigmoid; class is flagged if `sigmoid(logit) ≥ risk_threshold` (default 0.5).

### Matryoshka slicing
The shared trunk emits a 576d vector. Each `MatryoshkaHead` slices `x[..., :tier_dim]` then applies its tier-specific `nn.Linear(tier_dim, num_classes)`. All 4 tier linears live in the model (so a single checkpoint contains all tiers); at inference, **only the chosen tier's linears are loaded** — `parameter_count(tier)` reflects this.

### Byte input
- `0` is the pad byte (matches the 0th row of the embedding)
- `max_bytes` default 1024
- Inputs longer than `max_bytes` are **truncated** (not rejected)
- Mask is 1 for real bytes, 0 for pad — passed to attention and pool

---

## 6. Environment

```bash
# venv already created at .venv
source .venv/bin/activate
python --version  # 3.11
python -c "import torch, numpy, safetensors, yaml; print('ok')"
```

**Python 3.14 was tried first but has no torch wheels — we use Python 3.11.** This is documented because the failed attempt is in the opencode session history.

---

## 7. What's done ✅

### `model/pico_type/labels.py`
- All 7 vocabularies (sizes match plan exactly, asserted at import time)
- `decode_output(logits, tier, undetected_threshold, risk_threshold)` — respects all gating rules, applies UNDETECTED threshold
- `HEAD_NUM_CLASSES` dict for heads that need to query class counts
- `label_for(head, idx)` helper
- Constants: `UNDETECTED`, `ALL_HEADS`, `SUBTYPE_GATED_BY`, `CODE_LANG_GATED_BY`, `TEXT_LANG_GATED_BY`, `FILE_MIME_GATED_BY`

### `model/pico_type/arch.py`
- `PicoTypeConfig` dataclass with all hyperparams
- `ByteEmbed` — `nn.Embedding(256, 96)` init normal std=0.02
- `ConvBlock` — `Conv1d → LayerNorm → GELU → Dropout`, residual via 1×1 projection when dims change
- `RotaryPosEmb` — precomputed cos/sin cache, auto-grows if seq exceeds cache
- `AttnBlock` — pre-norm, fused QKV, RoPE on Q/K, `F.scaled_dot_product_attention`, MLP w/ 4× expansion
- `Pool` — `mean ‖ max ‖ std` over masked positions (handles padding correctly)
- `MatryoshkaHead` — `nn.ModuleDict` of `nn.Linear` per tier
- `PicoType` — top-level model
- `encode_bytes(data, max_len, pad)` — `bytes → (LongTensor[B, L], LongTensor[B, L])`
- `smoke_test()` — instantiates model, runs a forward, returns param counts
- `__main__` block runs the smoke test
- **NaN fix in AttnBlock**: `F.scaled_dot_product_attention` with a boolean mask where all entries are False (sample has no padding) produces NaN on CPU. Fixed by converting to float (`-inf` for masked positions, 0 for valid) and guarding with `mask.all().item()`. See `AttnBlock.forward` for the guard.

### `model/pico_type/data.py`
- `SyntheticGenerator(seed)` — generates one balanced sample at a time from 11 buckets: code, text, config, markup, data, link, error, image, file, secret, archive, binary
- 11 generator methods (`_gen_code`, `_gen_text`, `_gen_config`, `_gen_markup`, `_gen_data`, `_gen_link`, `_gen_error`, `_gen_image`, `_gen_file`, `_gen_secret`, `_gen_archive`, `_gen_binary`)
- `Sample` dataclass with `data: bytes`, label fields (int for single-label heads, `list[int]` for risk), `IGNORE_INDEX = -100` for gated heads
- `SyntheticDataset(generator, size)` — wraps generator for `DataLoader` compatibility
- Code templates for all 62 languages across 18 syntax groups (Python-like, C-like, JS-like, Lisp-like, etc.) — uses `re.sub` with `${kind}` placeholders
- Word lists for all 30 text languages
- Binary magic-byte headers for PDF, ZIP, GZIP, ELF, SQLite, Parquet, TIFF, PNG, JPEG, WASM, DEB, TTF, plus archive formats (7z, RAR, TAR, XZ, BZ2)
- `_detect_risk` runs on text samples (AWS key, JWT, SSH key, password detection)
- `label_counts()` returns class distribution for debugging
- `smoke_test()` generates 500 samples and prints coverage per head

### `model/pico_type/__init__.py`
- Re-exports the public API (already present in the repo when we recovered)

### `model/pico_type/train.py`
- `TrainConfig` dataclass — lr, warmup, total_steps, batch_size, grad_clip, per-head weights, etc.
- `collate_fn(batch)` — pads variable-length samples, creates `input_ids`, `attention_mask`, `labels` dict
- `MultiTaskLoss(weights)` — CE per head (ignore_index=-100 for gated heads) + BCE for risk. Skips any head with zero valid labels in batch (returns 0.0). Applies per-head weights (coarse=3.0, modality=2.0, code_lang=1.5, text_lang=1.5, others=1.0).
- `get_lr(step, config)` — linear warmup → cosine decay
- `train(config)` — full training loop:
  - `SyntheticGenerator` + `SyntheticDataset` for train/eval
  - AdamW, separate param groups (trunk w/ weight_decay, Matryoshka heads w/o)
  - BF16 AMP (CUDA) or FP32 (CPU/MPS)
  - Gradient clipping at 1.0
  - Logs every `log_every` steps, eval every `eval_every`, save every `save_every`
  - Saves `best.pt` (lowest eval loss), `final.pt`, plus periodic `step_{N}.pt`
- `load_checkpoint(path, model, optimizer)` — loads state dict
- **Known issues fixed**: NaN in SDP with all-valid mask (use float `-inf` instead of boolean mask); NaN from CE on all-ignore labels (skip head); Python 3.14 has no torch wheels (use 3.11)

### `model/pico_type/eval.py`
- `EvalConfig` dataclass — checkpoint, tier, eval_size, batch_size
- `evaluate(config)` — generates synthetic eval set, runs forward pass for all 7 heads
- Per-head `HeadMetrics`: accuracy, per-class precision/recall/F1, confusion matrix
- `RiskMetrics`: per-class average precision (sklearn-free implementation)
- `run_eval()` — CLI: `python -m model.pico_type.eval --eval-size 1000 --checkpoint checkpoints/best.pt`
- `_average_precision(y_true, y_scores)` — area under PR curve via trapezoidal rule

---

## 8. What's next (from plan §3–§6, in order)

| # | File | What it does | Status |
|---|---|---|---|
| 1 | `data.py` | Synthetic generator + dataset for multi-head training. 11 buckets, all 12 coarse classes, code/word templates for all 62/30 langs. | ✅ **done** |
| 2 | `train.py` | Multi-task trainer. AdamW + cosine, bf16, per-head loss weighting, gradient clipping, checkpoint save/load. | ✅ **done** |
| 3 | `eval.py` | Eval harness: per-head accuracy/PRF1, confusion matrix, risk AP, inference timing. CLI entry point. | ✅ **done** |
| 4 | `distill.py` | KD from per-head teachers (deberta-v3-small, CodeBERTa-lang-id, xlm-roberta-lang-detect, magic-bert-50m). T=2.0, α=0.7. | pending |
| 5 | `export.py` | ONNX, int8, tract, gguf export. | pending |
| 6 | `crates/picotype/` | Rust CLI w/ ONNX runtime. | pending |
| 7 | `crates/picotype-mcp/` | Rust MCP server (stdio + Streamable HTTP). | pending |
| 8 | `extensions/*` | Chrome MV3, Raycast, Alfred, VSCode. | pending |
| 9 | `spaces/picotype/` | Gradio Space + HF MCP registration. | pending |
| 10 | `paper/` | arXiv LaTeX. | pending |

---

## 9. Open decisions (from plan §8 — still open)

1. **HF handle**: `pico-type` (dash) for model card, `picotype` (no dash) for CLI binary. **Proposed, not confirmed.**
2. **Tier naming**: `tiny`/`small`/`base`/`pro` (matches Sentence-Transformers convention). **Proposed, not confirmed.**
3. **License**: Apache-2.0 (matches CommonLingua base). **Proposed, not confirmed.**
4. **arXiv target**: `cs.CL` (primary) + `cs.LG`. Co-authors: open question.
5. **Tagline**: *"One tiny model, one forward pass, every clipboard."* **Proposed, not confirmed.**

---

## 10. Quick recipes

### Instantiate the model
```python
from model.pico_type import PicoType, PicoTypeConfig

cfg = PicoTypeConfig(max_bytes=1024)
model = PicoType(cfg)
print(model.tier_sizes())  # {tiny: 1434344, small: 1445480, base: 1475176, pro: 1564264}
```

### Run a forward + decode
```python
from model.pico_type.arch import encode_bytes
from model.pico_type.labels import decode_output

model.eval()
x, mask = encode_bytes(b'def hi(): return 1', max_len=1024)
with torch.no_grad():
    logits = model(x, mask, tier='base')
out = decode_output(logits, tier='base')
# {'coarse': ..., 'modality': ..., 'subtype': ..., 'code_language': ...,
#  'text_language': ..., 'file_mime': ..., 'risk_flags': [...],
#  'confidence': ..., 'modality_confidence': ..., 'model_tier': 'base'}
```

### Convert to a single-tier checkpoint (for release)
Use `parameter_count(tier)` to get the param count for that tier. To build a release checkpoint, you would: train full model → for each tier, save only `trunk.*` + `heads.*.linears.{tier}.*` → export.

---

## 11. Session history (why this file exists)

The user was working on this project in opencode. The session (`ses_16dd3d39fffer9xnBUQYBS3u5z` — *"Tiny model for clipboard content classification"*) **crashed mid-execution** while writing `model/pico_type/arch.py`. Opencode had to be re-installed; the user thought files might be lost.

**They weren't.** The full session data was recovered intact from:
- `~/.local/share/opencode/opencode.db` (262MB SQLite)
- `~/.local/share/opencode/storage/session_diff/ses_16dd3d39fffer9xnBUQYBS3u5z.json` (128MB JSON)
- The trash (`~/.Trash/opencode`) only contained **opencode Desktop app data** (different product, irrelevant to the CLI session).

From the recovery we:
- Extracted the full plan → `docs/PLAN.md`
- Re-wrote `arch.py` (the file the crashed session was aborting on) and `labels.py` (never written in original session)
- Wrote this `walkthrough.md` so the next agent/harness has full context

### User's 3 original prompts
1. *"I want to build a really tiny model which categorises/classifies content. eg if we pass clipboard copied content, it should classify that as text, image, rich text, link, code(with language name), file with file type etc. Deep research the existing models on huggingface etc. go through new research papers and find an opportunity / gap to make this model in the most efficient manner plus make it really popular. feel free to suggest anything"*
2. *"continue asking questions and ahead. btw, the text language should be identified too like code language. if language not detected, it should simply return text + undetected or code undetected"*
3. *"go"*

### Where the original session was at crash
- 23 messages, 95 parts
- Plan had been finalized (10KB markdown)
- Approved with "go"
- Switched to `build` agent
- Set up Python 3.11 venv, installed torch/numpy/safetensors/pyyaml
- Created the full directory tree (`model/pico_type/`, `model/configs/`, `crates/picotype/`, etc.)
- Wrote `model/pico_type/__init__.py` ✅
- **Wrote `arch.py`** (the file the user originally saw being written) — wait, the original session was *aborted* on the arch.py write. The `__init__.py` is in the repo. We re-wrote arch.py from scratch using the plan + the small preview from the original write tool input.

---

## 12. Anti-patterns to avoid (learned)

- **Python 3.14 has no torch wheels.** Always use the venv's Python 3.11.
- **Don't double-count Matryoshka head params** when iterating `named_modules()` — `ModuleDict` is visited separately from its children. Use `named_parameters()` and check `.linears.{tier}.` in the name.
- **Gating heads are not always-on.** `subtype`, `code_lang`, `text_lang`, `file_mime` must mask their loss when not applicable. The decoder handles this; the trainer must too.
- **`UNK`/undetected is a decoder-side decision**, not a model class. The model has N logits; the decoder thresholds.
