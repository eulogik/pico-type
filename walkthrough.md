# pico-type — Walkthrough

> Living document. Every agent/harness picking up this project should start here. Also, you must keep updating as you go further.

---

## 1. What this is

**pico-type** is a tiny (~1.5M params), byte-level, multi-head content classifier. Input: up to 1024 raw bytes (clipboard text, file bytes, image header, etc.). Output: structured label set in one forward pass.

Built per the locked-in plan in [`docs/PLAN.md`](docs/PLAN.md) (recovered from the original opencode session — see §11).

### Why this exists
Existing clipboard tools are regex-only (ClipGate, 13 types) or LLM-powered (needs Ollama, GB-scale). Existing tiny classifiers do one job. **No model does all of them in one sub-5MB forward pass with a multi-head output.**

### Deliverables
- HuggingFace model (Apache-2.0) with 4 Matryoshka tiers ✅ (ONNX exported)
- Python CLI (`picotype`) ✅
- Gradio Space app ✅ (`gradio_app.py`)
- MCP server ✅ (`model/pico_type/mcp_server.py`)
- pytest smoke tests ✅ (`tests/test_smoke.py`)
- HF model card ✅ (`MODEL_CARD.md`)
- Badge'd README ✅
- Rust CLI (`crates/picotype/`) — pending
- Rust MCP server (`crates/picotype-mcp/`) — pending
- Browser extension, Raycast/Alfred/VSCode extensions — pending
- arXiv paper ✅ ([arXiv:2608.14658](https://arxiv.org/abs/2608.14658) — `paper/main.tex` — updated with v0.2 real-data results, PDF compiles)

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
├── .git/
├── checkpoints/                    # best.pt, ONNX models, eval results
├── docs/
│   └── PLAN.md                    # Full architecture plan
├── gradio_app.py                  # Gradio Space app
├── model/
│   └── pico_type/
│       ├── __init__.py            # re-exports public API
│       ├── labels.py              # vocabularies + decode_output
│       ├── arch.py                # PicoType model
│       ├── data.py                # synthetic generator
│       ├── train.py               # multi-task trainer
│       ├── eval.py                # eval harness
│       ├── distill.py             # KD pipeline
│       ├── export.py              # ONNX export
│       ├── cli.py                 # Python CLI (picotype)
│       └── mcp_server.py          # MCP server (stdio)
├── tests/
│   └── test_smoke.py              # pytest smoke tests
├── spaces/
│   └── requirements.txt           # HF Space dependencies
├── MODEL_CARD.md                  # HF model card
├── README.md                      # badge'd README
├── LICENSE                        # Apache-2.0
├── pyproject.toml
└── walkthrough.md                 # this file
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

### ARTH V2 Wk 1–2 kickoff (2026-09-22, branch `arth`)
- **Trunk freeze**: v0.2 torch weights survive nowhere (best.pt = pre-v2 95.2%; max ONNX-match 27/70) — recovered exact v0.2 trunk from single-file ONNX via `scripts/arth_recover_trunk.py` (fused transposes unfolded, 4 tier slices; torch↔ONNX max logit diff **1.3e-5**). Saved `checkpoints/arth_trunk_v02.pt` (gitignored, regenerable). Frozen trunk reproduces v0.2 ONNX **70/70** on the parity probe set.
- **Measured baselines** (`scripts/arth_probe.py` → `scripts/arth_baseline.json`, M4 10-core): trunk 1,430,400 params (embed 24,576 + conv 517,248 + attn 2×444,288); new BiAttn 444,288 (matches +0.4M claim); base P50 **33.4ms single-thread / 21.0ms default-threads** (README ~18ms is ~15% under measured — parity gate set to ≤21ms).
- **New heads wired** (`model/pico_type/arth.py`, all UNTRAINED): `SemanticHead` (~1.02M: byte-offset option markers → 2-layer pre-norm 192d transformer → 1 logit/option, K=2/4/10 verified) + `RelationalHead` (13 deterministic structural byte feats + linear) + `ActHead` + `Calibrator` (per mode/count-bucket temps, default 1.0). Torch e2e (trunk+stubs, 4-way) P50 **11.5ms** ≤ 21ms gate.
- **Tests**: `tests/test_arth.py` (8 tests: shapes, frozen grads, legacy parity, calibrator buckets, latency smoke) — **17/17 green** with smoke suite; ruff clean.
- **Demos run** (plumbing only, uniform outputs = untrained): semantic 4-way choice + relational balanced-vs-broken scoring.
- **Pre-existing inconsistency found (not fixed, out of scope)**: `run_torch` (short seq) vs `run_onnx` (padded-1024) disagree ≤0.1 prob on short inputs — conv has no mask so edge effects differ. Training batches are padded, so KD stays consistent.
- **Plan doc updated**: init source, confirmed param budget (+size warning: 4.6–5.4M → ~30–34MB FP32 possible over ≤27MB target), latency truth, all in `PICO-TYPE-V2-Breakthrough-Plan.md`.
- **Risk++ wired** (`model/pico_type/arth.py::RiskPlusPlusHead`): 14 labels (6 v0.2 overlaps first for alignment), Matryoshka slices per tier (238/910/2702/8078 params), 6 rows warm-started **bit-exact** from frozen trunk risk head (all tiers), 8 new rows random-init. `scripts/temperatures.json` (9 mode×bucket defaults @1.0). Viral-demo dry run: api_key=0.78 fires on AWS key (warm-start signal works); new labels spurious until trained (expected). Full e2e P50 11.6ms (≤21ms gate). **21/21 tests green** (4 new Risk++ tests: shapes, warm-start equality, AWS-key signal, temps loading).
- **Wk 3-4 training data** (`model/pico_type/arth_data.py`): 34,709 items, manifest frozen+committed at `data/arth_manifest.json` (corpora in gitignored `data/raw/arth/`). Splits: 4k Risk++ synth (8 generators × 500, 70/30 pos/neg) + 6k semantic choice + 2k score + 2k noul + 3k structural (bracket-surgery 4-way) + 2k invoice extraction + 2k unanswerable (evidence-stripped) + 8.7k Heap + 5k Wiki (hashed for provenance). Safety: all fake secrets EXAMPLE-marked or SSA-invalid (000/666/900+); audit found zero credential violations; attack payloads contain zero credential shapes. Determinism: cross-process verify PASS (caught + fixed a set-ordering bug in invoice distractors — `list(set)` → `sorted`). Missing-data loaders (Enron/ToxicChat/typed-decisions) skip gracefully with a message. 5 new data tests (generator validity, no-real-secrets, decision shapes, determinism, manifest+loaders). **26/26 green**, ruff clean.
- **Training loop** (`model/pico_type/arth_train.py`): mixed streams per step (8 legacy KD vs frozen teacher copy, temp 2.0 + 8 semantic choice/score/noul + 8 risk BCE-14 + 4 structural per-option-CE), loss = 1.0·KD + 1.0·sem + 0.5·rel + 0.5·risk + 0.005·calCE (uniform-if-wrong); Phase A trunk frozen → unfreeze @2000 steps with separate trunk LR group (3e-5 vs 3e-4); grad-clip 1.0, FP32 CPU, save/resume (`checkpoints_arth/`, gitignored weights). Smoke run 60 steps: total **1.96→0.64** (legacy≈0 frozen, all heads falling, no NaN); resume 50→70 verified; post-train legacy parity **70/70**. 2 new train tests (stream/collate shapes, 2-step finite + checkpoint). **28/28 green**, ruff clean.
- **Deep-verify audit** (pre-full-run gate): caught **4 real bugs** before they could poison training — (1) CRITICAL `pooled()` hard `no_grad` → post-unfreeze trunk got zero grad from sem/risk/rel (fixed; 33 trunk params now receive grads, proven); (2) CRITICAL resume crash post-unfreeze (param-group 1 vs 2 mismatch; optimizer now rebuilt from saved group structure); (3) HIGH lexicographic ckpt sort resumed from stale `arth_step_50` when `arth_step_100` existed (fixed numeric `latest_ckpt`); (4) HIGH test missing corpora skip → CI break. Hardening: offset-based relational grouping, dead code removed, `assert`→`FloatingPointError`, module-anchored data paths, heap/wiki roots in `verify_manifest` (PASS), structural attempt cap (rng-identical, manifest unchanged), trunk `.train()` on unfreeze, 0-step guard. Full re-verify: **32/32** (+4 regression tests), ruff clean, parity **70/70**, Phase-A losses **bit-identical** to original smoke (1.6183/1.3923 — refactor changed no math), freeze-boundary resume PASS, mid-Phase-B resume PASS, unfreeze fires (legacy KD 0→0.09 = trunk adapting under KD guard).
- **Full run (3000 steps)**: Phase A frozen 2000 + Phase B unfrozen 1000. Phase B **rejected by parity gate** (58/60 vs 60/60 at step_2000; head gains +0.3% choice — not worth regression). step_2000 kept as frozen-trunk base.
- **FP audit + hard-negative fix**: post-run benign FP 7/11 — generator negs were 5 reused templates. Added `benign_hard` split (1500 diverse + near-miss negatives: 'Please disregard my previous message…', 'Ignore the auto-formatter…'), manifest amended with original roots byte-locked. 3 Phase-A-only FT rounds → shipped step 3800.
- **Shipped ckpt `checkpoints_arth_ft/arth_final.pt`**: parity **60/60** (frozen trunk), choice **0.592**, noul 0.925, latency **5.2ms**, risk eval unseen seeds recall 0.81–1.00/spec 0.95–1.00. Thresholds: `scripts/risk_thresholds.json` via spec-constrained max-J + ε-floor for demo secret labels; `risk_flags()` API. Temps: ECE 0.018/0.023/0.060 (gate ≤0.10 ✓).
- **Known limitations (documented, not hidden)**: 3/11 worst-case benign probes still FP (misranked 0.82–0.98, threshold-immune); inject recall 0.81@spec0.95 (ROC overlap, not gated by plan); secrets_github row misses bare ghp demo (api_key row covers it). Random-benign FP ≈1–4.5%/label.
- **Tests: 34/34** (added: benign_hard, risk-thresholds/flags; calibrator asserts fitted temps), ruff clean. New scripts: `arth_fit_temps.py`, `arth_fit_thresholds.py`.
- **ToxicChat gate saga (2026-09-23)**:
  1. First eval **FAIL**: recall 0.582, AUROC 0.664 (kill <0.72) — synth templates don't transfer to real jailbreak style.
  2. Added `toxicchat_jail` TRAIN split (manifest amendment #2, original roots re-verified locked).
  3. **Leakage caught by new test**: train/test share 196 identical inputs (14/91 test positives) → first FT discarded; generator excludes test-seen inputs; retrained clean 600 steps from ft@3800.
  4. **Parity-protocol bug caught**: battery used unpadded inputs → bogus 54/60; ONNX protocol = fixed 1024-pad+mask → **60/60 all ckpts** (legacy heads inside frozen trunk — preds byte-identical across ckpts, proven).
  5. Gate metric corrected to selective accuracy @50% coverage (precision@flagged impossible at 1.8% base rate).
- **Shipped ckpt (new): `checkpoints_arth_ft2/arth_final.pt` (step 4400)**: ToxicChat **recall 0.824 / sel-acc@50 0.946 / AUROC 0.891 — PASS|PASS**; parity 60/60; choice 0.687; noul 0.854; latency 5.2ms; benign FP 2/11 (both inject); demos 7/7; ECE 0.020/0.027/0.050; temps 1.2605/0.4534/2.2513. `scripts/arth_eval_battery.py` = canonical battery. **35/35 tests, ruff clean.**
- **ONNX export of trained heads (2026-09-23)**: `python scripts/arth_export_onnx.py` → `checkpoints/arth_full_base.onnx` **11.66MB single-file IR8/opset18** — verify PASS: err 1.3e-5, ORT parity **60/60**, semantic **300/300**, risk flags **11/11**, ≤27MB gate. Bugs caught: torch2.12 dynamo default dropped weights to `.data` sidecar + broke MHA → forced TS exporter (`dynamo=False`) + inline save. **INT8 (4.29MB) decision-parity PASS** — plan's known quant crash absent on current stack; FP32 remains canonical ship. Protocol notes locked in verifier: fixed-1024 pad, trunk input = text only, options via opt_embs (training-consistent). **36/36 tests (new export smoke), ruff clean.**
- **RLCD-lite heads-only (2026-09-23)**: ES group-baseline (G=4, sigma 0.01, eta **2e-5** — 3e-4 collapsed reward, caught in smoke) on semantic+riskpp, 400 steps/26min, trunk byte-lock PASS. Shipped → **`checkpoints_arth_ft3/arth_final.pt` (step 4799)**: choice **0.737**, noul **0.929**, ToxicChat **0.846/0.970 PASS|PASS**, parity 60/60, FP 2/11, demos 7/7, ECE ≤0.10 (0.021/0.011/0.076). ONNX re-exported+verified from ft3 (11.66MB PASS). Nit: secrets_github floor T=0.011 double-flags bare AKIA (harmless). Hypercube 8k deferred (user policy). **37/37 tests, ruff clean.**

### Training, Deployment & Publishing
- **Training**: 1700 steps completed. eval_loss improved 6.33 (step 0) → 2.72 (step 800) → **1.97 (step 1700, best.pt)**. Subtype/code_lang accuracy dipped (overfitting), text_lang/risk improved.
- **v0.2 training** (real data): 6700 steps (1700 synthetic + 5000 mixed), best eval_loss **1.95** at step 6500. code_lang **60.3%** (The Heap), text_lang **98.2%** (Wikipedia).
- **ONNX export**: All 4 tiers re-exported from step 1700 best.pt (~200KB each, FP32, opset 18).
- **ONNX single-file + IR 8**: `scripts/make_single_onnx.py` merges external weights and lowers IR to 8 (onnxruntime-web WASM rejects IR ≥ 9). Identical outputs vs original (verified). Uploaded to `eulogik/pico-type-v02` (old `.onnx.data` files deleted; gradio app no longer downloads them).
- **HF Model**: `huggingface.co/eulogik/pico-type` — ONNX models + model card (updated after each training run).
- **HF Space**: `huggingface.co/spaces/eulogik/pico-type` — Gradio app fixed (self-contained, downloads ONNX from model repo at startup). **Label drift fixed**: gradio's label tables had drifted from the trained model (`file_mime` 88 vs 90, wrong order; `text_lang` tail wrong) — `scripts/gen_labels_artifacts.py` now generates `LABELS.py` + `workers/src/labels.js` from `model/pico_type/labels.py` (single source of truth). Space RUNNING, verified via gradio API.
- **Cloudflare Worker** (`workers/`): free HTTP API for all 4 tiers — onnxruntime-web 1.16.3 (WASM), single-file models fetched from HF with Cache API edge caching, CORS open. **Verified byte-identical to Python** (Node harness + parity check: all labels match, confidences within 0.002). Deploy: `npx wrangler login && npx wrangler deploy` (needs user's CF account). Caveats: local `wrangler dev` has no wasm on this Mac (workerd), and free CF plan ~10ms CPU quota is likely too small for ~100ms inference → paid $5/mo for reliable serving.
- **PyPI**: `pico-type` v0.1.0 published. v0.1.1 built (README fix) but not uploaded (file already exists error — version mismatch).
- **GitHub**: `github.com/eulogik/pico-type` — `main` branch + `v0.1` tag. CI passes (pytest + ruff).

### Known Training Issues
- **MPS OOM**: batch_size 64 causes MPS OOM (19+ GiB allocated). Fixed by reducing to batch_size=16 and `train_tiers=('base',)`.
- **MPS graph cache**: Writes to system `/tmp`, was filling disk when free space <1GB. ~9GB now available, OK.
- **Overfitting**: code_lang accuracy dropped 54%→42%, subtype 98%→94% from step 800 to 1700. May need more data diversity or lower LR.

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
|---|---|---|---|---|
| 1 | `data.py` | Synthetic generator + dataset for multi-head training. 11 buckets, all 12 coarse classes, code/word templates for all 62/30 langs. | ✅ **done** |
| 2 | `train.py` | Multi-task trainer. AdamW + cosine, bf16, per-head loss weighting, gradient clipping, checkpoint save/load. resume_from field for continuing training. | ✅ **done** |
| 3 | `eval.py` | Eval harness: per-head accuracy/PRF1, confusion matrix, risk AP, inference timing. CLI entry point. | ✅ **done** |
| 4 | `distill.py` | KD from per-head teachers (deberta-v3-small, CodeBERTa-lang-id, xlm-roberta-lang-detect). T=2.0, α=0.7. | ✅ **done** |
| 5 | `export.py` | ONNX export (opset 18), int8, tract, gguf. | ✅ **done** |
| 6 | `cli.py` | Python CLI (`picotype`) — stdin/file/clipboard input → ONNX inference → JSON output | ✅ **done** |
| 7 | `mcp_server.py` | MCP server (stdio transport) for Claude/Cursor/VSCode | ✅ **done** |
| 8 | `gradio_app.py` | Gradio Space app for HF Spaces | ✅ **done** |
| 9 | `tests/test_smoke.py` | pytest smoke tests (8 tests: arch, data, ONNX, CLI, labels) | ✅ **done** |
| 10 | `MODEL_CARD.md` | HuggingFace model card with eval results | ✅ **done** |
| 11 | `README.md` | Overhauled with badges, perf table, deploy links | ✅ **done** |
| 12 | `spaces/requirements.txt` | Dependencies for HF Space deployment | ✅ **done** |
| 13 | HF Model + Space | Published to huggingface.co/eulogik/pico-type (model) and /spaces/eulogik/pico-type (Space) | ✅ **done** |
| 14 | PyPI publish | pico-type v0.1.0 on PyPI (README not rendering; v0.1.1 built) | ✅ **done** |
| 15 | `crates/picotype/` | Rust CLI w/ ONNX runtime. | ✅ **done** |
| 16 | `crates/picotype-mcp/` | Rust MCP server (stdio + Streamable HTTP). | pending |
| 17 | `extensions/*` | Chrome MV3 scaffolded, Raycast, Alfred, VSCode. | pending |
| 18 | `workers/` | Cloudflare Worker HTTP API (free tier) — all 4 tiers, parity-verified vs Python. Deploy: `npx wrangler login && npx wrangler deploy`. | ✅ **done** (not yet deployed) |
| 18 | `paper/` | arXiv LaTeX (`paper/main.tex`) — updated to v0.2 with real-data results (code_lang 60.3%, text_lang 98.2%), 95% CIs, per-language tables, comparisons vs fastText/CLD2/Linguist/Pygments, data distribution, 20 references. PDF compiles clean. | ✅ **done** |
| 19 | Training | 6700 steps completed (1700 synthetic + 5000 mixed), best eval_loss 1.95 (step 6500), MPS (batch=16, base tier). Real data: 8709 code + 5000 text samples. ONNX exported to `eulogik/pico-type-v02`. | ✅ **done** |
| 20 | arXiv | Submitted Thu 30 Jul 2026 (cs.AI primary, cross-listed cs.CL/cs.CR/cs.IR/cs.LG, CC BY 4.0). | ✅ **done** |
| 21 | Docs correction | Size/latency claims fixed across paper (`main.tex` v2 draft), `README.md`, `MODEL_CARD.md`, `paper/v02_card.md`, HF org card, docs page: single-file FP32 sizes 9.09–9.61 MB (was "203–206 KB" graph-only); latency ~18 ms on M2 CPU (was "5.5–9.8 ms / <12 ms"); comparisons corrected (vs Linguist 1.6×, vs Pygments ~5×). HF main repo artifacts swapped to single-file IR-8, `.onnx.data` deleted; docs page 95.2% → 98.3%. | ✅ **done** |

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
