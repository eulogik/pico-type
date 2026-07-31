# pico-type Cloudflare Worker

Free (paid-tier optional) HTTP API for pico-type v0.2, serving the same 4 ONNX tiers
(tiny / small / base / pro) with byte-identical results to the Python runtime
(verified: all labels and confidences match `onnxruntime` within 0.002).

## API

- `GET /` — service info
- `POST /v1/classify` — `{"text": "...", "tier": "base"}` → per-head `label` + `confidence`,
  `risk` returns the full probability distribution.

```bash
curl -X POST https://pico-type.<your-subdomain>.workers.dev/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "def hello():\n    print('"'"'hi'"'"')", "tier": "base"}'
```

CORS is open (`*`) so it can be called from browsers.

## Deploy

```bash
npm install
npx wrangler login      # one-time, opens browser
npx wrangler deploy
```

## How it works

- onnxruntime-web 1.16.3, non-threaded `ort-wasm-simd.wasm` (fetched from jsDelivr
  on cold start; the wasm loader has no `wasmBinary` override in newer builds and
  **1.16.3 is the last version whose browser build honors `env.wasm.wasmBinary`,
  used only by the Node parity test**).
- Models are single-file ONNX (weights inlined), **IR version 8** — onnxruntime-web
  WASM builds reject newer IR versions. Regenerate with
  `../.venv/bin/python ../scripts/make_single_onnx.py`, then re-upload to
  `eulogik/pico-type-v02` on HF.
- On cold start the worker fetches the 9-10 MB model + 10 MB wasm (edge-cached via
  the Cache API), then compiles the session. Warm inference ≈ 100 ms; cold start ≈ 5-8 s.
- Labels are generated from `model/pico_type/labels.py` (single source of truth):
  `npm run gen-labels` → `src/labels.js`.

## Testing

```bash
node test/parity.mjs        # runs the real worker code in Node (caches shimmed)
../.venv/bin/python test/check_parity.py   # compares vs Python onnxruntime, expects "fails: 0"
```

## Caveats

- **Local dev has no wasm:** on some macOS setups, `wrangler dev` (workerd) disables
  wasm code generation ("Wasm code generation disallowed by embedder"). This is a
  local-runtime restriction; production Cloudflare Workers support wasm. Test with
  the Node harness above instead.
- **Free plan CPU quota:** the free Workers plan allows ~10 ms CPU per request, and
  one inference here is ~100 ms of wasm CPU time. Use the paid plan ($5/mo) for
  reliable serving, or the HF Space / HF Inference API for a zero-cost option.
