const tierSelect = document.getElementById("tier");
const refreshBtn = document.getElementById("refresh");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

async function classify(text) {
  const resp = await fetch("http://127.0.0.1:8130/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, tier: tierSelect.value }),
  });
  if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
  return resp.json();
}

function render(result) {
  resultsEl.innerHTML = "";
  const heads = ["coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime"];
  for (const head of heads) {
    const data = result[head];
    if (!data) continue;
    const div = document.createElement("div");
    div.className = "section";
    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.textContent = head.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase()) + ":";
    const valueEl = document.createElement("span");
    valueEl.className = "value";
    const badgeCls = { code: "code", secret: "secret", file: "file", archive: "file", image: "file", binary: "file" }[data.label] || "text";
    valueEl.innerHTML = `<span class="badge badge-${badgeCls}">${data.label}</span>`;
    const confEl = document.createElement("span");
    confEl.className = "confidence";
    confEl.textContent = ` (${(data.confidence * 100).toFixed(1)}%)`;
    div.appendChild(labelEl);
    div.appendChild(valueEl);
    div.appendChild(confEl);
    resultsEl.appendChild(div);
  }

  // Risk flags
  const risk = result.risk;
  if (risk) {
    const flagged = Object.entries(risk).filter(([, v]) => v > 0.5);
    if (flagged.length > 0) {
      const div = document.createElement("div");
      div.className = "section";
      const labelEl = document.createElement("span");
      labelEl.className = "label";
      labelEl.textContent = "Risk flags:";
      div.appendChild(labelEl);
      for (const [k, v] of flagged) {
        const el = document.createElement("div");
        el.className = "risk-item";
        el.textContent = `⚠ ${k} (${(v * 100).toFixed(0)}%)`;
        div.appendChild(el);
      }
      resultsEl.appendChild(div);
    }
  }
}

async function refresh() {
  try {
    statusEl.textContent = "Reading clipboard...";
    const text = await navigator.clipboard.readText();
    if (!text || !text.trim()) {
      statusEl.textContent = "Clipboard is empty";
      return;
    }
    statusEl.textContent = `Classifying ${text.length} chars...`;
    const result = await classify(text);
    statusEl.textContent = `Input: ${text.slice(0, 40)}${text.length > 40 ? "..." : ""}`;
    render(result);
  } catch (err) {
    statusEl.className = "error";
    statusEl.textContent = `Error: ${err.message}`;
  }
}

refreshBtn.addEventListener("click", refresh);
tierSelect.addEventListener("change", refresh);
document.addEventListener("DOMContentLoaded", refresh);
// Auto-refresh every 2s
setInterval(refresh, 2000);
