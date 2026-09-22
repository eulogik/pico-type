# PICO-TYPE V2 "ARTH" — The Semantic Byte Guardrail

> **One-liner:** pico-type's ~18ms CPU body (V2 target ≤8ms) + Laya's semantic brain + RLCD calibration + relational reasoning. Fixed taxonomy AND arbitrary schema + structural reasoning in one forward pass. The edge guardrail every agent needs.
>
> **Status:** BREAKTHROUGH PLAN — Sept 22, 2026 (incorporating all papers through Sept 19)
> **Scope:** standalone, in-repo. Build on an `arth` branch of `eulogik/pico-type` (author-owned: GitHub/HF/PyPI/paper — no fork, no upstream negotiation; ship as v0.3, merge yourself). Never compete with NIRNAY on long-context semantics — own the CPU-rate lane.
> **Name:** ARTH (अर्थ = meaning). Keep `picotype` PyPI + `/v1/systemone` compat (Jev-type typed-decision route — request/response spec in NIRNAY plan). Publish as `picotype[semantic]` extra, merge to main yourself.

---

## 0. Where pico-type v0.2 stands (facts)

* 1.43–1.56M, `ByteEmbed→Conv(k3,5,7)→2×BiAttn(RoPE)→Pool(mean‖max‖std)→576d → 7 Matryoshka heads (16/64/192/576)`. ≤1024 raw bytes, no tokenizer.
* 9MB ONNX FP32 (INT8 export blocked on multi-head shape-inference — ~1.4MB is an estimate, unregister as a fact), **~18ms CPU base (README, verified)** — per-tier 3/5/12ms claims not confirmed this pass. Paper's "under 210KB / under 10ms" = graph-only file / tiny tier, not the full base artifact. CLI/MCP/Rust/WASM/Gradio, Apache-2.0, M4-trained.
* Heads: coarse 12 / modality 8 / subtype 24 / code 62 / text 30 / MIME 90 / risk 6 (sigmoid). Gated loss+decoder, `undetected` @ max<0.4, risk @ ≥0.5.
* Real: code 60.3% Heap (+57pp over synth-only), text-lang 98.3% Wiki (+79pp), synth heads 100%, real 20/21 (95.2%). Weak: 8 code langs <50%, Malay 70%, >1024B trunc, YAML→error, no semantics/calibration/NLP.
* Sources: github.com/eulogik/pico-type, HF eulogik/pico-type*, arXiv:2608.14658, eulogik.com/lab.

**The gap nobody owns:** CPU-rate (V2 target ≤8ms; v0.2 ~18ms today, no GPU) + arbitrary semantic decisions (choice/score/noul) + calibrated probs + structural reasoning + secrets/PII/jailbreak in one pass. pico-type has lane 1, Laya/Jev have lane 2–3 on GPU/API. V2 takes all three at 1.5–6M params.

---

## 1. V2 architecture (frozen — incorporates Sept papers)

```
Bytes ≤8k (patch-chunked: 1k windows + coding-rate boundaries + attention-pool, else trunc — document choice)
  → v0.2 trunk FROZEN init (1.5M) 
       + 1× BiAttn layer (192d, +0.4M) for semantic context
       + Relational Attention head (DAT-style, +0.2M) for structural patterns
  → shared 576d trunk
  → legacy 7 heads (unchanged, zero regression)  // v0.2 parity gate
  ＋ semantic head (new, +2.5M):
       per-option [MASK]-style byte markers (like Laya, but byte offsets not token ids)
       → 2-layer pre-norm Transformer (192d) → scorer → 1 logit/option
       → Choice softmax | Score RPS-expectation | Noul sigmoid
       → Relational scorer (structural: AST patterns / invoice fields / logic forms)
       → temp per (type, count) → calibrated dist
       → Act/Escalate (pooled + [top1, margin, entropy, k/255])
  → Risk++ head (extends risk 6 → 14: + prompt_injection, jailbreak, pii_ssn, pii_card, secrets_aws, secrets_github, sql_injection, xss_payload)
  → Matryoshka slices for semantic+relational heads too (16/64/192/576) — tiny tier answers in ~4ms
  → Hypercube sparse wiring for 8k context (fixed sparse rotated, 1/32 links, log₂n layers — no training overhead)
```

**Total: ~4.5M params (base) → ONNX ≤27MB FP32 / ≤6MB INT8 single file (from v0.2's measured ~6 B/param FP32, ~1 B/param INT8; 18MB raw-weight floor — earlier 12MB/3MB violated physics, fixed 2026-09-22). Param budget CONFIRMED Wk 1–2 (`scripts/arth_probe.py`): frozen trunk 1,430,400 (embed 24,576 + conv 517,248 + attn 2×444,288 + pool 0) + new BiAttn 444,288 (=+0.4M claim ✓) → new trunk ≈1.87M; + semantic scorer ~2–2.5M + relational 0.2–0.5M + Risk++/act ~0.5M → total ≈4.6–5.4M. Watch: high end lands ~30–34MB FP32, over the ≤27MB target — size gate re-checked once heads are wired. Edge artifact is tiny-tier INT8 (~1.5M params → ~2MB). P50 target: ≤8ms base CPU / ≤4ms tiny (INT8+fused; speedup budget below).**

**Why breakthrough (not incremental):**

1. **First byte-level semantic decider.** No BPE → no Khmer/Hindi shred (Laya 0%@0.95 failure gone by construction), no vocab files, binary-safe (scan executables/zips/headers Jev/Laya can't ingest).
2. **Fixed + semantic + structural in one pass.** Gate: if `coarse==code` skip semantic NLI, run relational (AST patterns); if `risk≥0.5` force escalate. Single 8ms call replaces pico (~18ms today → ≤8ms in V2) + Laya (32ms GPU) chain for 80% of triage.
3. **RLCD-lite at 4M scale.** Reward = Brier+correctness (bounded, RLCR) + per-bucket temps. Kathleen proved 469K no-pretrain 92.4% AG News — architecture > pretrain at this scale. We prove calibration transfers down.
4. **Guardrail-native.** Risk++ (14) + act head + relational = the MCP middleware agents need: block secret exfil / prompt-inject / binary-exec / SQLi / XSS pre-tool-call, log distribution + threshold. Jev/Laya are too heavy for per-tool-call gating; regex (ClipGate 13 types) too dumb.
5. **8k via hypercube + coding-rate patching, honest trunc.** Hypercube wiring (Sept 16): fixed sparse rotated reaches all positions in log₂n layers, 1/32 links, 2.4x faster **at 8k (~parity at ≤1k — it buys context, not 1k latency)**. Coding-rate boundaries (ByteFlow) > entropy for semantic chunking. Publish trunc-vs-patch ablation (LargitData lesson).
6. **Speedup budget (18ms → ≤8ms, settled by Wk 1–2 probe):** INT8 ~1.5–2× + fused single-file export ~1.1–1.2× + Matryoshka tier-routing (easy inputs exit tiny/small) ≈ 2.4× combined → ~7.5ms. Thin margin by design — if the probe shows >10ms, the §6 fallback fires in Wk 2, not Wk 6 (cut attn layer / distill to linear scorer).
7. **Relational Attention (DAT, Sept 17):** separates sensory vs structural attention → better data efficiency & OOD generalization. Critical for code/invoice/logic patterns where structure matters more than surface form.

**Explicit non-goals:** 77-way single-shot (use coarse-to-fine: pico-coarse → top-20 → semantic choose; Laya lesson), generation, multilingual NLI beyond 30 text-langs v1 (mmBERT lane belongs to NIRNAY), GPU requirement (CPU-only is the moat).

---

## 2. Training (3h, M4 or T4, reproducible)

* **Init:** recovered v0.2 trunk `checkpoints/arth_trunk_v02.pt` frozen 2k steps → unfreeze all. Semantic+relational heads from scratch (like Laya decision head). NOTE 2026-09-22: `checkpoints/best.pt` is the pre-v2 95.2% checkpoint (max_bytes=2048) and v0.2 torch weights survive nowhere on disk/HF/backup — trunk was recovered by ONNX→torch weight transfer from the single-file v2 exports (fused transposes unfolded, tier slices restored; torch↔ONNX max logit diff 1.3e-5 on padded inputs, `scripts/arth_recover_trunk.py`). No local checkpoint matches v2 ONNX (best 27/70 head-matches).
* **Data (~25k decisions):** synth templates (pico `data.py` 11 buckets) + real (Heap code, Wiki langs, Enron spam, ToxicChat jailbreak, BigBench-inject, PII synthetic vault, SQLi/XSS payloads — never real secrets) + typed-decisions train subset (400 cases, to prove transfer) + unanswerable pairs (evidence-stripped, Kev pattern) + structural reasoning pairs (code AST diffs, invoice field extractions, logic form transforms). Freeze hashes pre-train. Enron/ToxicChat are public datasets — verify redistribution terms before shipping derived artifacts; never ship PII-adjacent rows.
* **Loss:** `L = 1.0*L_legacy7 (frozen-target KD, no regression) + 1.0*L_sem (CE/RPS/BCE) + 0.5*L_rel (structural CE) + 0.5*L_risk++ (BCE) + λ*L_calCE (uniform-if-wrong, λ=0.005)`. Phase B RL 30min: Gaussian noise + Brier reward + group baseline.
* **Calibrate:** temps per (choice/score/noul × 2/3-5/6-20) on held-out. Ship `temperatures.json` + `refit.py`. Report raw+fitted always.
* **Export:** ONNX opset 18, dynamic axes, INT8 first (fix multi-head shape-inference — known pending; **fallback: per-head export, else FP16**, latency budget re-checked either way), WASM + Rust verify.

---

## 3. Benchmarks (pre-registered)

| Suite | V2 target | v0.2 / rivals |
|---|---|---|
| Legacy 7 parity | **zero regression** (coarse ≥99.5%, text-lang ≥98%, code ≥60%) | v0.2 numbers above |
| AG News / SST-2 / Enron spam | ≥0.92 / ≥0.84 / ≥0.98 | Kathleen 92.4% AG (469K); Laya AG 0.953 / spam 0.993 (in-training) |
| ToxicChat jailbreak held-out | ≥0.78, @50% cov ≥0.93 | Laya 0.755–0.762 / 0.931 |
| Prompt-inject (BigBench-inject synth) | ≥0.88 P/R | regex ~0.6, LLM ~0.9 @ 500ms (we win on rate) |
| Semantic 4-way/10-way (own split) | ≥0.88 / ≥0.70 held-out, no train-overlap | Laya-ft 0.766 (fine-tuned — we report train/test split honestly) |
| Risk++ secrets/PII/SQLi/XSS | AP ≥0.97 synth, ≥0.88 real-world hand-set 100 items | v0.2 1.0 synth only |
| Relational structural (code AST / invoice / logic) | ≥0.85 4-way, ≥0.70 10-way | N/A (new) |
| Latency CPU (M4/Ryzen, ONNX) | P50 1Q ≤8ms base / ≤4ms tiny; 10Q batch ≤30ms | v0.2 ~18ms (README); Laya 32ms GPU; Jev 250ms API |
| Size | ≤27MB FP32 / ≤6MB INT8 single file (base); tiny INT8 ~2MB | v0.2 9MB FP32 / ~1.4MB INT8 |
| ECE fitted | ≤0.08 | Laya 0.081, Jev ~0.10–0.24 |

Rules: per-item logs, CPU model + threads pinned, raw+fitted columns, trunc documented, pytest+ruff green (v0.2.2 baseline extended with semantic-head + latency-smoke tests).

*CPU-trade note: AG 0.92 vs Laya 0.953 and SST-2 0.84 are deliberate CPU-rate trades (4.5M CPU vs 421M GPU) — closing them is NIRNAY's job, not V2's. Enron ≥0.98 ≈ parity with Laya's in-training 0.993 on held-out terms. ToxicChat ≥0.78 and semantic 0.88 target above Laya; misses are caught by §6 gates.*

---

## 4. Viral GTM (edge devs, not LLM Twitter)

1. **`echo "sk-ant-..." | picotype --pretty` moment.** Risk++ demo: paste AWS key / inject string / SQLi / XSS → instant red flag offline (V2 target ≤8ms; v0.2 ~18ms). Regex can't, LLM too slow. Clip it.
2. **MCP one-liner:** `picotype-mcp` in Claude/Cursor/VSCode: every tool-call gated pre-exec. "The sub-10ms bouncer for your agent." Jev can't sit here (API latency + $).
3. **Browser/WASM demo:** `eulogik.github.io/pico-type/demo.html` → add semantic box: type any rule ("is this a refund?") + threshold slider + live distribution. No key, no GPU.
4. **Chrome/Raycast/VSCode extensions (finish pending scaffolds):** clipboard → coarse+MIME+risk+semantic+relational in <10ms. "Every paste, typed."
5. **Kaggle + Space:** fine-tune notebook (free T4, 3h, own labels → temps) + HF Space with 14 risk flags live. Laya won goodwill this way; edge lane is emptier.
6. **In-repo launch (you are upstream):** build on the `arth` branch, publish as `picotype[semantic]` extra (`pip install "picotype[semantic]"`), merge to main as v0.3 yourself. No fork-war possible — spend the goodwill budget on docs and demos instead. You already own the standard.

---

## 5. Build (45 days, solo)

* **Wk 1–2:** branch (not fork), freeze trunk, wire semantic markers + relational head + temps + act head. Confirm §1 param budget + CI green (pytest+ruff extended). **Latency smoke test: 1024B dense vs hypercube, 1Q P50 — measured 2026-09-22 (M4 10-core, `scripts/arth_probe.py`): base P50 33.4ms single-thread / 21.0ms default-threads. README's ~18ms is ~15% under measured — gate: hypercube ≤ dense AND base ≤21ms (no regression vs measured stock v0.2, not the aspirational 18ms), with heads stubbed.** Gate: legacy parity + latency probe + 1 semantic demo + 1 relational demo.
* **Wk 3–4:** Risk++ data + RLCD-lite + per-bucket temps + hypercube wiring for 8k. Gate: ToxicChat ≥0.72, ECE fitted ≤0.10, 8k context no quality drop.
* **Wk 5–6:** INT8 (or FP16/per-head fallback per §2)/WASM/Rust, MCP, Space, extensions, docs, V2 paper draft (benchmarks table is paper-shaped), merge to main as v0.3. Gate: all §3 or publish miss.

**Cost:** $0–$50 (M4 local + free Kaggle). No GPU bill excuse.

---

## 6. Kill gates

* Legacy regression >1pt → revert, semantic as sidecar only.
* Semantic <0.80 4-way after RLCD → cut arbitrary schema, ship Risk++ guardrail only (still viral).
* ToxicChat <0.72 after RLCD → cut semantic claims, ship Risk++ guardrail only (still viral).
* CPU P50 >10ms base → cut extra attn layer, distill to linear scorer.
* Relational <0.70 4-way → ship without, keep Risk++.
* Never: train on real user secrets, claim zero-shot = fine-tuned, compare CPU-ms to API-ms unlabeled.

---

## 7. How V2 + NIRNAY fit (no cannibalism — the stack)

```
Paste / file / tool-call / ticket / clipboard
  → ARTH v2 (≤8ms CPU target, offline): coarse/MIME/risk + first semantic cut + relational + abstain?
      → confident → act (no GPU, no $)
      → unsure → NIRNAY-1 (35ms GPU): concepts + MoM memory + halting + full schema + structural
          → unsure → LLM (seconds): prose/reasoning/generation
```

V2 is the bouncer. NIRNAY is the judge. LLM is the author. Sell the stack, not the parts: `pip install nirnay[edge]` bundles both, one `Router()` picks tier by latency budget (`--budget 10ms|50ms|500ms`).

---

### Sources

eulogik/pico-type GitHub/HF/PyPI, arXiv:2608.14658, eulogik.com/lab, Laya HF/site/BENCHMARKS, Kathleen arXiv:2604.07969, BLT arXiv:2412.09871 / BLT-D 2605.08044 / MBP 2608.15454, ByteFlow 2603.03583, Carpathian byte-LM, UTF-8 validity 2606.14122, hierarchy-ablation 2609.00463, **Hypercube wiring arXiv:2609.18145**, **Relational Attention / DAT arXiv:2609.20530**, **SGDR arXiv:2609.22884**, RLCR 2507.16806, ConfTuner 2508.18847, Rewarding Doubt ICLR 2026, CALM ACL 2026 findings-acl.610, MoM arXiv:2502.13685, ConvMem 2609.10441.

*Amendment log: 2026-09-22 v2 breakthrough plan incorporating all Sept papers. In-repo build (author-owned); no fork.*
*Wk 1–2 progress 2026-09-22 (branch `arth`): trunk recovered from ONNX (`arth_trunk_v02.pt`, 1.3e-5 verified); semantic/relational/act/calibrator stubs wired (`model/pico_type/arth.py`, ~1.02M trainable, all UNTRAINED); frozen-trunk legacy parity 70/70 vs v2 ONNX; torch e2e (trunk+stubs, 4-way) P50 11.5ms ≤ 21ms gate; 17/17 tests green (`tests/test_arth.py`); semantic + relational demos run (uniform outputs = untrained, plumbing only). Risk++ wired: 14-label Matryoshka head, 6 overlaps warm-started bit-exact from v0.2 risk rows (all tiers), 8 new rows random; `scripts/temperatures.json` defaults (all 1.0); viral-demo dry run shows api_key=0.78 on AWS key (warm-start signal real, new rows noise until Wk 3-4); full e2e P50 11.6ms; 21/21 tests green. Wk 3-4 data DONE (`model/pico_type/arth_data.py`): 34,709 items frozen in `data/arth_manifest.json` (21k new decisions: 4k Risk++ synth 8×500 + 6k choice + 2k score + 2k noul + 3k structural + 2k invoice + 2k unanswerable; 13.7k real Heap/Wiki hashed for provenance); all fake secrets EXAMPLE-marked/SSA-invalid (audit: zero credential violations); cross-process determinism verified (PYTHONHASHSEED 1/99 PASS — caught a set-ordering bug in invoice distractors); Enron/ToxicChat/typed-decisions loaders skip gracefully (no datasets lib, files absent); 26/26 tests green. Next: full training run + hypercube 8k + RLCD-lite + temps refit.*
*Training loop DONE 2026-09-22 (`model/pico_type/arth_train.py`): mixed streams (8 legacy KD temp 2.0 + 8 semantic + 8 risk + 4 relational), L = 1.0·KD + 1.0·sem + 0.5·rel + 0.5·risk + 0.005·calCE; Phase A frozen → unfreeze @2000 (trunk LR 3e-5); smoke 60 steps total 1.96→0.64, no NaN, resume 50→70 verified, post-train legacy parity 70/70; 28/28 tests green.*
*Deep-verify audit 2026-09-22 (pre-full-run): caught 4 real bugs — (1) CRITICAL: `ArthModel.pooled()` hard-wrapped `no_grad`, so post-unfreeze the trunk received ZERO gradient from sem/risk/rel paths (only KD trained it); fixed, Phase-B proof: 33 trunk params now receive grads. (2) CRITICAL: resume crashed post-unfreeze (`load_state_dict` param-group mismatch 1 vs 2); fixed — optimizer built from saved group structure, saved group count is source of truth. (3) HIGH: checkpoint resume used lexicographic sort (`arth_step_100` < `arth_step_50`) → stale resume; fixed numeric `latest_ckpt()`. (4) HIGH: `test_train_losses_finite` had no corpora skip → CI break; fixed. Also: dead opt_owner removed, rigid j*4 relational slicing → offset grouping, bare assert → FloatingPointError, arth_data paths CWD→module-anchored, verify_manifest now covers heap/wiki roots (PASS 8.4s), structural generator 50-attempt cap (rng-identical on success path → manifest unchanged), 0-step main() guard, trunk.train() on unfreeze. Re-verified: 32/32 tests (4 new: pooled-grad-flow, numeric-ckpt-sort, resume-across-unfreeze, verify-manifest), ruff clean, legacy parity 70/70, Phase-A smoke losses bit-identical (1.6183/1.3923), freeze-boundary resume + mid-Phase-B resume both PASS, unfreeze fires correctly (legacy KD rises 0→0.09 as trunk adapts under guard).*
*Full training + Wk3-4 calibration DONE 2026-09-22 (shipped ckpt `checkpoints_arth_ft/arth_final.pt`, step 3800, gitignored/regenerable): original 3000-step run → Phase B REJECTED (trunk unfreeze @2000 bought +0.3% choice/+0.2% risk but cost legacy parity 60/60→58/60 on close probes — plan gate 'no regression' fails; Phase A end step_2000 kept as trunk-frozen base). FP audit after run: 7/11 benign false positives — root cause: generator negs were 5 reused templates. Fix: `benign_hard` split (1500 diverse + 8 near-miss hard negatives, all-zero risk14), manifest AMENDED with original 8 synth + heap/wiki roots byte-locked (verified identical), benign×1 risk-pool mix. 3 Phase-A-only FT rounds (2000→3800, trunk frozen ⇒ parity structurally guaranteed). Shipped numbers: legacy parity **60/60**, choice acc **0.592** (uniform 0.25), noul abstain 0.925, e2e latency **5.2ms** (≤21 gate), risk eval on unseen seeds: recall 0.81–1.00 / spec 0.95–1.00 (AUC 0.96–1.00). Risk++ per-label thresholds fitted (`scripts/arth_fit_thresholds.py` → `risk_thresholds.json`, spec-constrained max-J, ε-floor only for demo-critical secret labels; `risk_flags()` in arth.py): all demos fire (bare AKIA→api_key, wrapped→+secrets_aws, ghp→api_key, jailbreak/inject/sqli/xss positives ✓). KNOWN LIMITATION (honest): 3/11 hand-picked benign probes still FP (meeting-notes→inject 0.82, 'plain benign'→jailbreak 0.93, SQL→jailbreak 0.98) — genuinely misranked, threshold-immune; random-benign FP rate 0.9–4.5%/label (eval spec ≥0.95). inject is the weak label (recall 0.81@spec 0.95, ROC overlap real). Temps refit on shipped ckpt (`arth_fit_temps.py`): ECE choice 0.018 / noul 0.023 / score 0.060 — **all ≤0.10 gate** (≤0.08 target met for choice+noul). **34/34 tests, ruff clean.** Next: RLCD-lite Phase B (reward=Brier+correctness, guarded), hypercube 8k wiring, ToxicChat eval when data available, ONNX export of trained heads.*
*Fixes 2026-09-22 (external review, double-checked): size targets → physics (≤27MB FP32 / ≤6MB INT8 + param budget); explicit speedup budget + Wk 1–2 latency probe; owner framing (arth branch, merge yourself); INT8 fallback (per-head/FP16); ToxicChat kill gate; CI + V2 paper + dataset-terms + /v1/systemone spec notes; "210KB = graph-only" clarification; CPU-trade note on below-Laya targets.*
*Verification 2026-09-22: pico latency corrected to ~18ms (GitHub README; earlier 3/5/12ms and 5ms claims unconfirmed); text-lang 98.3% (README); all cited arXiv abstracts verified this pass.*