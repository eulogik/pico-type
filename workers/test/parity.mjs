import { readFileSync } from "node:fs";
import * as ortNS from "onnxruntime-web";

const ort = ortNS.default ?? ortNS;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmBinary = readFileSync(
  new URL("../node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm", import.meta.url)
);

globalThis.caches = {
  default: {
    async match() { return undefined; },
    async put() {},
  },
};

const { default: worker } = await import("../src/index.js");

const ENV = {
  MODEL_BASE_URL: "https://huggingface.co/eulogik/pico-type-v02/resolve/main",
  ORT_WASM_URL: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort-wasm-simd.wasm",
  DEFAULT_TIER: "base",
};

async function callWorker(path, body) {
  const req = new Request(`http://localhost${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const resp = await worker.fetch(req, ENV);
  return { status: resp.status, body: await resp.json() };
}

const SAMPLES = [
  { text: "def hello():\n    print('Hello, world!')", tier: "base" },
  { text: "The quick brown fox jumps over the lazy dog.", tier: "base" },
  { text: "<html><body><h1>Welcome</h1></body></html>", tier: "base" },
  { text: '{\n  "name": "pico-type",\n  "version": "0.2.0"\n}', tier: "base" },
  { text: "#!/usr/bin/env python3\nimport os\nprint('hello')", tier: "base" },
  { text: "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR", tier: "base" },
  { text: "sk-9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0", tier: "base" },
  { text: "{\"a\":1,\"b\":[true,null,2.5]}", tier: "tiny" },
  { text: "¡Hola! ¿Cómo estás? Estoy muy bien, gracias.", tier: "small" },
  { text: "SELECT * FROM users WHERE id = 42;", tier: "pro" },
  { text: "", tier: "base" },
];

const results = {};
for (let i = 0; i < SAMPLES.length; i++) {
  const s = SAMPLES[i];
  const r = await callWorker("/v1/classify", s);
  results[`${i}|${s.tier}`] = r;
  console.log(`${r.status} ${r.body?.error ?? "ok"} #${i} tier=${s.tier} latency=${r.body?.latency_ms ?? "-"}ms`);
}

const fs = await import("node:fs");
fs.writeFileSync("/tmp/worker_results.json", JSON.stringify(results, null, 1));

const info = await callWorker("/");
console.log("GET / status:", info.status, "tiers:", info.body.tiers);
