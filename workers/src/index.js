import * as ortNS from "onnxruntime-web";
import { ALL_HEADS, LABEL_TABLES } from "./labels.js";

const ort = ortNS.default ?? ortNS;

const MAX_BYTES = 1024;
const CACHE_PREFIX = "https://picotype.dev/model/";

let ENV = { MODEL_BASE_URL: "", ORT_WASM_URL: "", DEFAULT_TIER: "base" };

const sessions = new Map();
let wasmInitPromise = null;

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  const out = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

async function fetchCached(url, cacheKey) {
  const cache = caches.default;
  const hit = await cache.match(cacheKey);
  if (hit) return hit;
  const resp = await fetch(url);
  if (resp.ok) {
    try {
      await cache.put(cacheKey, resp.clone());
    } catch (err) {
      console.warn("cache.put failed:", err.message);
    }
  }
  return resp;
}

async function ensureWasm() {
  if (!wasmInitPromise) {
    wasmInitPromise = (async () => {
      ort.env.wasm.numThreads = 1;
      if (!ort.env.wasm.wasmBinary) {
        const wasmUrl = new URL(ENV.ORT_WASM_URL);
        ort.env.wasm.wasmPaths = wasmUrl.origin + wasmUrl.pathname.slice(0, -"/ort-wasm-simd.wasm".length);
      }
    })();
  }
  return wasmInitPromise;
}

function getSession(tier) {
  if (!sessions.has(tier)) {
    sessions.set(
      tier,
      (async () => {
        await ensureWasm();
        const url = `${ENV.MODEL_BASE_URL}/picotype_${tier}.onnx`;
        const resp = await fetchCached(url, CACHE_PREFIX + `picotype_${tier}.onnx`);
        if (!resp.ok) throw new Error(`failed to fetch model: ${resp.status}`);
        const bytes = new Uint8Array(await resp.arrayBuffer());
        return ort.InferenceSession.create(bytes, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      })()
    );
  }
  return sessions.get(tier);
}

function tokenize(text) {
  const bytes = new TextEncoder().encode(text).subarray(0, MAX_BYTES);
  const ids = new BigInt64Array(MAX_BYTES);
  const mask = new Uint8Array(MAX_BYTES);
  for (let i = 0; i < bytes.length; i++) {
    ids[i] = BigInt(bytes[i]);
    mask[i] = 1;
  }
  return { ids, mask };
}

async function classify(text, tier) {
  if (!text) throw new Error("empty text");
  const session = await getSession(tier);
  const { ids, mask } = tokenize(text);
  const results = await session.run({
    input_ids: new ort.Tensor("int64", ids, [1, MAX_BYTES]),
    attention_mask: new ort.Tensor("bool", mask, [1, MAX_BYTES]),
  });
  const out = {};
  for (const head of ALL_HEADS) {
    const probs = softmax(results[head].data);
    if (head === "risk") {
      out[head] = Object.fromEntries(
        LABEL_TABLES[head].map((label, i) => [label, Number(probs[i].toFixed(4))])
      );
    } else {
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      out[head] = { label: LABEL_TABLES[head][best], confidence: Number(probs[best].toFixed(4)) };
    }
  }
  return out;
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

export default {
  async fetch(request, env) {
    ENV = env;
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    const url = new URL(request.url);

    if (url.pathname === "/" && request.method === "GET") {
      return json({
        name: "pico-type",
        version: "0.2.0",
        description: "Byte-level multi-head content classifier (coarse, modality, subtype, code_lang, text_lang, file_mime, risk).",
        tiers: ["tiny", "small", "base", "pro"],
        max_input_bytes: MAX_BYTES,
        model: ENV.MODEL_BASE_URL,
        usage: { post: { path: "/v1/classify", body: { text: "content to classify", tier: "base" } } },
      });
    }

    if (url.pathname === "/v1/classify" && request.method === "POST") {
      try {
        const body = await request.json();
        const text = typeof body.text === "string" ? body.text : "";
        const tier = ["tiny", "small", "base", "pro"].includes(body.tier) ? body.tier : ENV.DEFAULT_TIER;
        if (!text) return json({ error: "text is required" }, 400);
        const start = Date.now();
        const result = await classify(text, tier);
        return json({ tier, latency_ms: Date.now() - start, ...result });
      } catch (err) {
        return json({ error: err.message || String(err) }, 500);
      }
    }

    return json({ error: "not found" }, 404);
  },
};
