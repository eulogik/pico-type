// pico-type content script
// Shows a small badge next to text selections

let badge = null;

function createBadge() {
  badge = document.createElement("div");
  badge.id = "picotype-badge";
  badge.style.cssText = `
    position: fixed;
    bottom: 8px;
    right: 8px;
    background: #1a1a1a;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font: 11px -apple-system, system-ui, sans-serif;
    z-index: 2147483647;
    display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  `;
  document.body.appendChild(badge);
}

document.addEventListener("mouseup", () => {
  const sel = window.getSelection().toString().trim();
  if (sel.length > 10) {
    if (!badge) createBadge();
    badge.textContent = "pico-type...";
    badge.style.display = "block";
    chrome.runtime.sendMessage(
      { action: "classify", text: sel, tier: "base" },
      (response) => {
        if (response?.success) {
          const r = response.result;
          const cls = r.coarse?.label || "?";
          const conf = r.coarse?.confidence || 0;
          badge.textContent = `${cls} (${(conf * 100).toFixed(0)}%)`;
        } else {
          badge.textContent = "pico-type: err";
        }
        setTimeout(() => { badge.style.display = "none"; }, 3000);
      }
    );
  }
});
