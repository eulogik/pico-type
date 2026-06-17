"""Local HTTP server for Chrome extension inference."""

from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = "127.0.0.1"
PORT = 8130

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
from model.pico_type.cli import load_onnx_model, run_onnx  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    session = None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
            text = req.get("text", "")
            result = run_onnx(Handler.session, text)
            result["text_length"] = len(text)
            resp = json.dumps(result, ensure_ascii=False).encode()
        except Exception as e:
            resp = json.dumps({"error": str(e)}, ensure_ascii=False).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):
        pass


def main():
    model_dir = os.environ.get("PICOTYPE_MODEL_DIR", "checkpoints")
    tier = os.environ.get("PICOTYPE_TIER", "base")
    path = os.path.join(os.path.dirname(__file__), "../..", model_dir)
    Handler.session = load_onnx_model(tier, os.path.abspath(path))
    server = HTTPServer((HOST, PORT), Handler)
    print(f"pico-type server running on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
