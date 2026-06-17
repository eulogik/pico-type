"""pico-type MCP server (stdio transport) for Claude/Cursor/VSCode."""

from __future__ import annotations

import json
import os
import sys

from .cli import load_onnx_model, run_onnx


def handle_request(req: dict, session) -> dict:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "tools": {
                        "classify": {
                            "description": "Classify content type, language, modality, and risk",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "Content to classify"},
                                    "tier": {"type": "string", "enum": ["tiny", "small", "base", "pro"], "default": "base"},
                                },
                                "required": ["text"],
                            },
                        },
                        "classify_file": {
                            "description": "Classify a file's content type and risk",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "description": "File path"},
                                    "tier": {"type": "string", "enum": ["tiny", "small", "base", "pro"], "default": "base"},
                                },
                                "required": ["path"],
                            },
                        },
                    },
                },
            },
        }

    if method == "tools/call":
        tool = req.get("params", {}).get("name", "")
        args = req.get("params", {}).get("arguments", {})

        if tool == "classify":
            text = args.get("text", "")
            result = run_onnx(session, text)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "json", "json": result}]}}

        if tool == "classify_file":
            path = args.get("path", "")
            if not os.path.exists(path):
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": f"File not found: {path}"}}
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            result = run_onnx(session, text)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "json", "json": result}]}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
    model_dir = os.environ.get("PICOTYPE_MODEL_DIR", "checkpoints")
    tier = os.environ.get("PICOTYPE_TIER", "base")
    session = load_onnx_model(tier, model_dir)

    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req, session)
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            err = {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
