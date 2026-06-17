// pico-type background service worker
// Monitors clipboard updates and maintains classifier state

const CLASSIFY_URL = "http://127.0.0.1:8130/classify";

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ tier: "base" });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "classify") {
    fetch(CLASSIFY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: request.text, tier: request.tier || "base" }),
    })
      .then(r => r.json())
      .then(result => sendResponse({ success: true, result }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }
});
