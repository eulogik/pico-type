# Final Plan: **pico-type** — A 3.5M Multi-Head Byte Classifier for Universal Content Type Detection

## 1. Executive Summary

**pico-type** is a tiny, fully open-source, byte-level multi-head classifier that takes any blob (clipboard text, file bytes, image header, etc.) and emits a structured label set in one forward pass. It ships as:

- A HuggingFace model (Apache-2.0) with multi-tier Matryoshka exports
- A Rust + Python CLI (`picotype`)
- A Gradio Space that doubles as an **MCP server** (callable from Claude, Cursor, VSCode)
- A browser extension (compete with ClipGate)
- A Raycast/Alfred extension
- A companion arXiv paper

**Why now / the gap:** Existing clipboard tools are either regex-only (ClipGate, 13 types) or LLM-powered (Clipboard_ai, needs Ollama, GB-scale). Existing tiny classifiers do one job (code-language, MIME, intent, PII). No model does **all of them in one sub-5MB forward pass with a multi-head output**.

## 2. Architecture

```
Inputs (≤1024 UTF-8 bytes, masked/padded)
  │
  ├─ ByteEmbed (256 → 96d, learned; optional FFT-Rotate variant from Kathleen)
  │
  ├─ 3× Conv1D block (kernel 3,5,7)  + GELU + residual
  │
  ├─ 2× Bidirectional attention block  (d=192, 4 heads, RoPE θ=500k)
  │     [ablation: also test a Kathleen-style O(L) oscillator trunk
  │      for the sub-1MB mobile tier]
  │
  ├─ Pool = [mean ‖ max ‖ std]  → 576d shared trunk
  │
  └─ Matryoshka heads (independent Linear at 16/64/192/576 dims)
       ├─ h_coarse        (12-way softmax)   — primary type
       ├─ h_modality      (8-way softmax)    — textual / binary-image / binary-archive …
       ├─ h_subtype       (24-way softmax)   — JSON, YAML, CSV, HTML, … (only if coarse∈{config,markup,data})
       ├─ h_code_lang     (62-way softmax + "undetected")  — if coarse=code
       ├─ h_text_lang     (30-way softmax + "undetected")  — if coarse=text
       ├─ h_file_mime     (90-way softmax + "undetected")  — if modality∈{binary-*}
       └─ h_risk          (6-way sigmoid multi-label)      — api_key, jwt, ssh_key, password, email, phone
```

**Output schema (always returned, with confidence):**
```json
{
  "coarse":  "code",
  "modality":"textual",
  "subtype": null,
  "code_language":    "python",        // or "undetected"
  "text_language":    null,
  "file_mime":        null,
  "risk_flags":       [],
  "confidence":       0.94,
  "model_tier":       "standard"       // tiny | small | standard | full
}
```

**Size targets (Matryoshka tiering):**
| Tier | Dim | Params | FP32 | INT8 | Target device |
|---|---|---|---|---|---|
| `pico-type-tiny` | 16 | ~0.5M | 2 MB | 0.5 MB | MCU / IoT |
| `pico-type-small` | 64 | ~1.5M | 6 MB | 1.5 MB | Browser (WASM), mobile |
| `pico-type-base` | 192 | ~3.5M | 14 MB | 3.5 MB | Desktop CLI, browser ext |
| `pico-type-pro` | 576 | ~8M | 32 MB | 8 MB | Server / accurate use |

## 3. Training

**Framework:** Pure PyTorch (~600 LoC train script), no transformers dependency. Permissive.

**Distillation teachers (Polymorph-style per-head):**
- `microsoft/deberta-v3-small` (140M) → coarse + modality
- `huggingface/CodeBERTa-language-id` (84M) → code_lang
- `papluca/xlm-roberta-base-language-detection` (270M) → text_lang
- `mjbommar/magic-bert-50m-roformer-classification` (42M) → file_mime
- T=2.0, α=0.7 KD loss, plus hard-label CE on 30% of batch

**Data sources (all open / free):**
- Code: `bigcode/the-stack-v2` (multilingual), `cakiki/rosetta-code`
- Natural text: Wikipedia dumps, OSCAR, mC4 (subset)
- Configs/markup: GitHub `*.{json,yaml,toml,ini,xml,html,md,tex}` via BigQuery GH archive
- URLs: CommonCrawl WAT
- Errors: StackOverflow / GH issue bodies (regex-filtered for traceback patterns)
- File bytes: `mjbommar/binary-tokenizer-001-64k` corpora, plus synthetic magic-byte headers
- Images: 1 KB header slices from OpenImages V7
- Secrets: synthetic generation (regex + entropy filter), `gptmail-secret-detection` datasets

**Synthetic data generator:** 1 Python script producing balanced mixtures of: prose paragraphs, code snippets (multi-lang, snippets of 64–1023 bytes), config files, error traces, etc. Critical for head coverage.

**Training recipe:**
- Sequence length 1024 bytes
- AdamW, peak lr 3e-3, cosine, 5% warmup, 30 epochs
- bf16, batch 128, grad clip 1.0
- Task-balanced sampling: each head gets ≥1 sample per batch
- Loss = Σ_head w_h · L_h  with w tuned on val

**Hardware:** Single A100 80GB, ~36 h total training (3 seeds).

## 4. Data → Label Pipeline

| Bucket | Source | Auto-labels |
|---|---|---|
| Code | The Stack v2 | `language` field, repo paths |
| Prose | Wikipedia, mC4 | `lang` metadata |
| Config/markup | GH files | file ext + content sniff |
| URL | CommonCrawl WAT | regex for `http(s)://` |
| Error | SO, GH issues | regex for `Traceback`, `Error:`, `at line` |
| File bytes | binary-tokenizer corpora | magic-byte first-32B lookup |
| Image header | OpenImages | first 32B → format (PNG/JPEG/WebP/…) |
| Secret | synthetic | regex + entropy ≥ 4.5 |

## 5. Distribution & Popularity

### 5.1 HuggingFace deliverables
- `pico-type/pico-type-base` — main model card with eval harness, ONNX int8/fp16, tract-onnx, GGUF
- `pico-type/pico-type-tiny` … `-pro` — Matryoshka tiers
- `pico-type/picotype-space` — Gradio Space (also a **registered MCP server** on HF Hub — free distribution via `huggingface.co/mcp` proxy)
- `pico-type/picotype-eval` — public eval suite (reproducible JSON benchmarks)
- Datasets: `pico-type/synth-clipboard-v1`, `pico-type/eval-suite-v1`

### 5.2 Software deliverables
- **CLI** (`picotype`): Rust binary, ONNX runtime, sub-5ms inference. Reads stdin/file/clipboard, prints JSON.
- **MCP server** (`picotype-mcp`): stdio + Streamable HTTP, exposes tools: `classify`, `classify_batch`, `watch_clipboard` (macOS/Windows), `classify_history`
- **Browser extension** (Chrome/Firefox): replaces ClipGate's regex with pico-type; per-type icons; on-device, 100% local
- **Raycast extension** + Alfred workflow
- **VSCode extension**: paste-with-type, status bar shows type of selection
- **iOS Shortcut / Android Tasker** plug-in

### 5.3 Launch timeline (4–6 weeks)
| Week | Milestone |
|---|---|
| 1 | Repo scaffold, byte-level pipeline working on synthetic data, all heads wired |
| 2 | Real data ingest, distillation teachers downloaded, base tier trained, eval harness |
| 3 | All four Matryoshka tiers trained, ONNX/tract/GGUF exported, ablations complete |
| 4 | CLI (Rust), MCP server (Python+Gradio), HF Space + MCP registration, model card |
| 5 | Browser extension (MV3), Raycast/VSCode extensions, docs site, demo video |
| 6 | arXiv preprint, Show HN, r/MachineLearning, r/LocalLLaMA, r/programming launches |

### 5.4 Growth hooks
- First **MCP-native content classifier** — every Claude/Cursor/VSCode user is a potential user
- Free-tier on HF = automatic inference endpoint exposure
- Trending strategy: launch on a Tuesday, cross-post 4-5 channels same day, follow-up "vs ClipGate benchmarks" post
- Maintain a "What's new in pico-type" weekly digest → builds audience
- Open evaluation suite → community contributes new types (e.g., `meme_url`, `arxiv_id`, `semantic_version`) via a small "head-add" fine-tuning recipe
- Roadmap V2: adding a user-contributed `custom_types` head (LoRA per type) à la Polymorph

## 6. Repo structure (proposed)

```
pico-type/
├── README.md
├── LICENSE                          # Apache-2.0
├── model/
│   ├── pico_type/
│   │   ├── arch.py                  # ByteHybrid trunk + heads
│   │   ├── train.py                 # multi-task trainer
│   │   ├── distill.py               # KD from per-head teachers
│   │   ├── data.py                  # synthetic generator + dataset
│   │   ├── export.py                # ONNX, int8, tract, gguf
│   │   └── eval.py                  # public eval harness
│   └── configs/                     # tier configs
├── crates/picotype/                 # Rust CLI
├── crates/picotype-mcp/             # Rust MCP server (or Python)
├── extensions/
│   ├── chrome/                      # MV3
│   ├── raycast/
│   ├── alfred/
│   └── vscode/
├── paper/                           # arXiv LaTeX
├── spaces/picotype/                 # Gradio + MCP
└── docs/                            # mintlify or mkdocs
```

## 7. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Multi-head gradients fight (task interference) | Gradient norm clipping per-head, gating + scheduled dropout |
| 30 langs underperform on low-resource | Reuse PleIAs/CommonLingua approach, weight by data |
| Byte-level convs slow on long inputs | Cap at 1024 bytes, document limit, Matryoshka tiers |
| MCP ecosystem churns | Support stdio + Streamable HTTP (latest spec), drop SSE |
| Competition clones us | First-mover + multi-channel launch + arXiv paper + community eval suite |

## 8. Open decisions before I start coding

1. **Name lock**: `pico-type` confirmed. Handle: `pico-type` on HF? Or `picotype` (no dash)? I suggest `pico-type` for HF and `picotype` for the CLI to match existing naming patterns.
2. **Tier naming**: I propose `tiny`/`small`/`base`/`pro` matching Sentence-Transformers conventions.
3. **OSI license**: Apache-2.0 (matches CommonLingua base, patent grant). Confirm.
4. **arXiv target**: `cs.CL` (primary) + `cs.LG`. Co-authors: open question.
5. **Tagline** (for the model card): "One tiny model, one forward pass, every clipboard."

---

**Ready to execute when you give the green light.** I'll start by scaffolding the repo (`model/pico_type/arch.py` + `data.py` + minimal `train.py`) and getting a single-head baseline running on synthetic data — proving the byte-level pipeline + multi-head design before we invest in full data ingest. Estimated time-to-first-baseline: ~2 hours.
