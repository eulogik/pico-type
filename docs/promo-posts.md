# pico-type Promotional Posts

---

## 1. r/MachineLearning

**Title:** `[D] pico-type: 1.5M-param byte-level multi-head classifier – seeking feedback on architecture decisions`

**Body:**

I've been working on pico-type, a lightweight content classifier that operates directly on raw UTF-8 bytes with no tokenizer. The model has ~1.5M parameters, runs inference in <6ms on CPU, and exports to a ~9MB ONNX file.

**Key design choices I'd love feedback on:**

**Byte-level approach:** The model takes raw bytes (0-255) through an embedding layer, then 3 parallel Conv1D kernels (k=3,5,7) for local patterns, followed by 2 bidirectional attention layers with RoPE. No subword vocabulary, no pretrained embeddings. This lets it handle all 62+ languages and binary content equally, but I'm wondering if there are fundamental limits to what byte-level features can capture vs. token-level representations.

**Multi-head training:** 7 independent classification heads (coarse type, modality, subtype, code language, text language, MIME type, risk flags) share a single trunk. I'm using gated heads – e.g., code_lang only activates when coarse=code. Loss weighting was empirical: `coarse=8.0, modality=2.0, code_lang=2.0, text_lang=1.5, others=1.0`. The risk head uses BCE since it's multi-label. Is there a more principled approach to multi-task loss balancing?

**Matryoshka quantization:** 4 tier slices (16d/64d/192d/576d) share the same trunk, with separate linear heads per tier. This gives a 4x speed-accuracy tradeoff at inference with zero overhead to switch. Curious if others have explored this for multi-head architectures specifically.

**Results:** 95.2% accuracy on 21 hand-curated real-world inputs. 100% on modality, text language, code language. The one failure case is YAML configs being misclassified as errors at short input lengths – a byte-level ambiguity I haven't solved.

**Links:**
- GitHub: https://github.com/eulogik/pico-type
- PyPI: `pip install pico-type`
- Browser demo (runs entirely client-side): https://eulogik.github.io/pico-type/demo.html
- ONNX model: https://huggingface.co/eulogik/pico-type

Would appreciate thoughts on the architecture, especially the byte-level vs. token-level tradeoffs and multi-task loss strategies.

---

## 2. r/LocalLLaMA

**Title:** `I trained a 1.5M param model that classifies content from raw bytes – no tokenizer needed, runs in browser`

**Body:**

Hey everyone, wanted to share something I've been building. It's called pico-type – a tiny content classifier that works directly on raw UTF-8 bytes.

**Why it's interesting for local inference:**
- **1.5M parameters**, ~9MB ONNX file
- **<6ms inference** on CPU (no GPU required)
- **No tokenizer** – feeds raw bytes directly into the model
- **7 classification heads** in a single forward pass: content type, modality, 62 code languages, 30 text languages, 24 subtypes, 90 MIME types, risk flags (API keys, JWTs, passwords)
- **Runs in browser** via ONNX Runtime Web – try the demo, zero server needed

**The architecture:**
```
Bytes → Embedding(256→96d) → 3×Conv1D(k=3,5,7) → 2×BiAttention(RoPE) → Pool → 7×Matryoshka Heads
```

**Matryoshka tiers** let you trade accuracy for speed:
- tiny (16d): ~3ms, 8.7MB
- base (192d): ~5ms, 8.8MB  
- pro (576d): ~12ms, 9.1MB

All tiers share the same weights – switch at inference with zero overhead.

**Quick start:**
```bash
pip install pico-type
echo "def hello(): pass" | picotype --pretty
```

Trained on a mix of synthetic data + ~650 real GitHub files across 62 languages. Apache 2.0, fully open.

**Links:**
- GitHub: https://github.com/eulogik/pico-type
- PyPI: https://pypi.org/project/pico-type/
- HuggingFace: https://huggingface.co/eulogik/pico-type
- Demo: https://eulogik.github.io/pico-type/demo.html

---

## 3. r/Python

**Title:** `Show HN: pico-type – a 1.5M-param content classifier you can install with pip`

**Body:**

I built pico-type, a lightweight content classifier for Python. It identifies what type of content you're dealing with – code, text, images, config files, secrets – directly from raw bytes.

**Install and use in 30 seconds:**

```bash
pip install picotype
echo '{"name": "test"}' | picotype --pretty
```

**What it classifies (7 heads, one pass):**
- Content type: code, text, image, config, markup, data, error, secret, archive, binary
- Code language: 62 languages (Python, JS, TypeScript, Java, C, Go, Rust, SQL, etc.)
- Text language: 30 languages
- Subtype: JSON, YAML, HTML, Markdown, SQL, Dockerfile, etc.
- MIME type: 90 types
- Risk flags: API keys, JWTs, passwords, emails, SSH keys
- Modality: textual vs binary variants

**Why it's useful:**
- Clipboard monitoring (classifies whatever you copy)
- Data pipeline routing (auto-sort files by type)
- Security scanning (detects leaked secrets)
- Language detection for code snippets

**Technical details:**
- ~1.5M parameters, ~9MB ONNX file
- <6ms inference on CPU
- No tokenizer – works on raw UTF-8 bytes
- 4 Matryoshka tiers for speed/accuracy tradeoff
- Apache 2.0 license

**Links:**
- GitHub: https://github.com/eulogik/pico-type
- PyPI: https://pypi.org/project/pico-type/
- Browser demo: https://eulogik.github.io/pico-type/demo.html

---

## 4. Twitter/X Thread

**Tweet 1:**
I trained a 1.5M parameter model that classifies content from raw bytes – no tokenizer needed.

62 code languages, 30 text languages, 90 MIME types, risk flag detection – all in <6ms on CPU.

Try the browser demo (runs entirely client-side): https://eulogik.github.io/pico-type/demo.html

**Tweet 2:**
Architecture: Bytes → 3 parallel Conv1D (k=3,5,7) → 2 BiAttention layers with RoPE → pooling → 7 Matryoshka heads.

4 tiers (16d→576d) share one trunk. Switch accuracy/speed at inference with zero overhead.

~9MB ONNX, Apache 2.0.

**Tweet 3:**
Links:
📦 PyPI: pip install pico-type
🐙 GitHub: https://github.com/eulogik/pico-type
🤗 HuggingFace: https://huggingface.co/eulogik/pico-type
🌐 Demo: https://eulogik.github.io/pico-type/demo.html

Built by @eulogik. Feedback welcome.

---

## 5. Hacker News (Show HN)

**Title:** `Show HN: Byte-level content classifier – detects 62 languages from raw bytes, no tokenizer, 95% accuracy`

**Body:**

Hi HN, I built pico-type – a 1.5M-parameter model that classifies content directly from raw UTF-8 bytes. No tokenizer, no subword vocabulary, no pretrained embeddings.

It runs inference in <6ms on CPU, exports to a ~9MB ONNX file, and runs entirely in the browser via ONNX Runtime Web. You can try it here: https://eulogik.github.io/pico-type/demo.html

**What it does:** One forward pass produces 7 classification heads simultaneously:
- Content type (code/text/image/config/etc.)
- Code language (62 languages)
- Text language (30 languages)
- Subtype (JSON, YAML, HTML, Markdown...)
- MIME type (90 types)
- Risk flags (API keys, JWTs, passwords)
- Modality (textual vs binary)

**Quick try:**
```bash
pip install picotype
echo "def hello(): pass" | picotype --pretty
```

**Three questions I'd love feedback on:**

1. **Byte-level vs token-level:** This model operates on raw bytes (0-255) rather than tokens. It handles all languages and binary content equally, but I'm wondering if there are fundamental limits to what byte-level features can capture. Has anyone explored this tradeoff systematically?

2. **Multi-task loss balancing:** The 7 heads use weighted loss (coarse=8.0, modality=2.0, code_lang=2.0, text_lang=1.5, others=1.0). I arrived at these weights empirically. Is there a more principled approach for multi-head architectures with different label distributions?

3. **Matryoshka for multi-head:** I use 4 tier slices (16d/64d/192d/576d) that share the trunk but have separate linear heads. This gives a clean speed-accuracy tradeoff. Has anyone seen this applied to multi-head classifiers specifically?

**Links:**
- GitHub: https://github.com/eulogik/pico-type
- PyPI: https://pypi.org/project/pico-type/
- HuggingFace: https://huggingface.co/eulogik/pico-type
- Browser demo: https://eulogik.github.io/pico-type/demo.html
