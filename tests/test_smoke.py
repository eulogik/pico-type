"""pico-type smoke tests: architecture, data, ONNX inference, CLI, MCP."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from model.pico_type.arch import PicoType, PicoTypeConfig, TIERS
from model.pico_type.labels import ALL_HEADS, COARSE_LABELS, MODALITY_LABELS, CODE_LANG_LABELS, TEXT_LANG_LABELS, FILE_MIME_LABELS, RISK_LABELS, HEAD_NUM_CLASSES
from model.pico_type.data import SyntheticGenerator, SyntheticDataset
from model.pico_type.cli import load_onnx_model, run_onnx


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "checkpoints", "best.pt")
ONNX_DIR = ROOT


@pytest.fixture(scope="session")
def model():
    cfg = PicoTypeConfig(max_bytes=128)
    m = PicoType(cfg)
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.eval()
    return m


def test_architecture_params():
    for tier in TIERS:
        cfg = PicoTypeConfig(max_bytes=128)
        m = PicoType(cfg)
        count = m.parameter_count(tier)
        assert 1_400_000 <= count <= 1_600_000, f"{tier}: params={count}"


def test_architecture_forward():
    cfg = PicoTypeConfig(max_bytes=128)
    m = PicoType(cfg).eval()
    ids = torch.randint(0, 256, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.bool)
    out = m(ids, mask)
    for head in ALL_HEADS:
        logits = out[head]
        assert logits.shape[0] == 2, f"{head} batch dim"
        assert logits.shape[1] == HEAD_NUM_CLASSES[head], f"{head} num classes"


def test_labels_completeness():
    assert len(COARSE_LABELS) == 12
    assert len(MODALITY_LABELS) == 8
    assert len(CODE_LANG_LABELS) == 62
    assert len(TEXT_LANG_LABELS) == 30
    assert len(FILE_MIME_LABELS) == 90
    assert len(RISK_LABELS) == 6
    assert len(ALL_HEADS) == 7


def test_synthetic_generator():
    gen = SyntheticGenerator(seed=42)
    sample = gen()
    assert isinstance(sample.data, bytes)
    assert len(sample.data) > 0
    assert 0 <= sample.coarse < 12


def test_synthetic_dataset():
    ds = SyntheticDataset(SyntheticGenerator(seed=42), 100)
    assert len(ds) == 100
    sample = ds[0]
    assert len(sample.data) > 0


@pytest.mark.skipif(not os.path.exists(os.path.join(ONNX_DIR, "checkpoints", "picotype_base.onnx")), reason="ONNX model not found")
def test_onnx_inference():
    session = load_onnx_model("base", os.path.join(ONNX_DIR, "checkpoints"))
    text = "def hello(): pass"
    result = run_onnx(session, text)
    for head in ["coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime"]:
        assert head in result, f"missing {head}"
        assert "label" in result[head], f"{head} missing label"
        assert "confidence" in result[head], f"{head} missing confidence"
    assert "risk" in result
    assert len(result["risk"]) > 0


@pytest.mark.skipif(not os.path.exists(os.path.join(ONNX_DIR, "checkpoints", "picotype_base.onnx")), reason="ONNX model not found")
def test_cli_stdin():
    result = subprocess.run(
        [sys.executable, "-m", "model.pico_type.cli", "--tier", "base"],
        input=b"print('hello')",
        capture_output=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    out = json.loads(result.stdout)
    assert "coarse" in out


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="checkpoint not found")
def test_model_loads():
    cfg = PicoTypeConfig(max_bytes=128)
    m = PicoType(cfg)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    assert m is not None
