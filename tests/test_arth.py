"""ARTH Wk 1-2 tests: head shapes, frozen-trunk parity, calibrator, latency smoke."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import torch

from model.pico_type.arth import (
    N_STRUCT_FEATS,
    ArthModel,
    Calibrator,
    bucket_count,
    dist_stats,
    mark_options,
    structural_features,
)
from model.pico_type.labels import (
    ALL_HEADS,
    COARSE_LABELS,
    CODE_LANG_LABELS,
    FILE_MIME_LABELS,
    MODALITY_LABELS,
    SUBTYPE_LABELS,
    TEXT_LANG_LABELS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUNK = os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt")
BASELINE = os.path.join(ROOT, "scripts", "arth_baseline.json")


@pytest.fixture(scope="module")
def arth():
    if not os.path.exists(TRUNK):
        pytest.skip("recovered trunk not found")
    m = ArthModel(TRUNK)
    m.eval()
    return m


def _ids_mask(data: bytes):
    ids = torch.tensor([[b for b in data[:1024]]])
    return ids, torch.ones_like(ids, dtype=torch.bool)


def test_mark_options_spans():
    joined, spans = mark_options(b"input", [b"opt0", b"opt1"])
    assert joined == b"input\x00opt0\x00opt1"
    assert spans == [(6, 10), (11, 15)]
    assert joined[spans[0][0] : spans[0][1]] == b"opt0"
    assert joined[spans[1][0] : spans[1][1]] == b"opt1"


def test_semantic_shapes(arth):
    pooled = torch.randn(2, 576)
    for k in (2, 4, 10):
        out = arth.semantic(pooled, torch.randn(2, k, 96))
        assert out.shape == (2, k), out.shape


def test_relational_act_shapes(arth):
    pooled = torch.randn(2, 576)
    r = arth.relational(pooled, torch.randn(2, N_STRUCT_FEATS))
    assert r.shape == (2,), r.shape
    a = arth.act(pooled, torch.randn(2, 4))
    assert a.shape == (2, 2), a.shape


def test_trunk_frozen(arth):
    assert all(not p.requires_grad for p in arth.trunk.parameters())
    assert any(p.requires_grad for p in arth.semantic.parameters())


def test_legacy_parity(arth):
    lt = {
        "coarse": COARSE_LABELS,
        "modality": MODALITY_LABELS,
        "subtype": SUBTYPE_LABELS,
        "code_lang": CODE_LANG_LABELS,
        "text_lang": TEXT_LANG_LABELS,
        "file_mime": FILE_MIME_LABELS,
    }
    with open(BASELINE) as f:
        ref = json.load(f)["parity_onnx"]
    import sys as _sys

    _sys.path.insert(0, ROOT)
    from scripts.arth_probe import PROBE_SET

    probes = {n: d for n, d in PROBE_SET if n in ("py_code", "png")}
    for name, data in probes.items():
        ids = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        n = len(ids)
        pad = np.zeros(1024, dtype=np.int64)
        pad[:n] = ids
        mask = np.zeros(1024, dtype=bool)
        mask[:n] = True
        r = arth.legacy(torch.tensor([list(pad)]), torch.tensor(mask[None, :]))
        key = name
        for h in ALL_HEADS:
            if h == "risk":
                continue
            assert lt[h][int(r[h][0].argmax())] == ref[key][h], (name, h)


def test_riskpp_shapes_and_labels(arth):
    from model.pico_type.arth import RISK_PLUS_LABELS
    from model.pico_type.labels import RISK_LABELS

    assert len(RISK_PLUS_LABELS) == 14
    assert RISK_PLUS_LABELS[:6] == list(RISK_LABELS)
    pooled = torch.randn(2, 576)
    for tier in ("tiny", "small", "base", "pro"):
        out = arth.riskpp(pooled, tier)
        assert out.shape == (2, 14), (tier, out.shape)


def test_riskpp_warm_start(arth):
    for tier in ("tiny", "small", "base", "pro"):
        a = arth.riskpp.head.linears[tier].weight[:6]
        b = arth.trunk.heads["risk"].linears[tier].weight
        assert torch.equal(a, b), tier
        ba = arth.riskpp.head.linears[tier].bias[:6]
        bb = arth.trunk.heads["risk"].linears[tier].bias
        assert torch.equal(ba, bb), tier


def test_riskpp_aws_key_signal(arth):
    ids, mask = _ids_mask(b"aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
    p = arth.pooled(ids, mask)
    from model.pico_type.arth import RISK_PLUS_LABELS

    probs = torch.sigmoid(arth.riskpp(p)).tolist()[0]
    assert probs[RISK_PLUS_LABELS.index("api_key")] > 0.5


def test_calibrator_loads_defaults():
    from model.pico_type.arth import Calibrator

    c = Calibrator(os.path.join(ROOT, "scripts", "temperatures.json"))
    assert c.temp("choice", 4) == 1.0
    assert c.temp("noul", 15) == 1.0


def test_calibrator_buckets():
    c = Calibrator()
    assert c.temp("choice", 4) == 1.0
    assert bucket_count(2) == "2-2"
    assert bucket_count(4) == "3-5"
    assert bucket_count(10) == "6-20"
    logits = torch.zeros(1, 4)
    assert torch.allclose(c.apply(logits, "choice"), torch.full((1, 4), 0.25))
    st = dist_stats(torch.tensor([[0.5, 0.25, 0.125, 0.125]]))
    assert st.shape == (1, 4)
    assert abs(st[0][0].item() - 0.5) < 1e-6
    assert abs(st[0][1].item() - 0.25) < 1e-6


def test_structural_features_deterministic():
    a = structural_features(b"def f(x):\n  return [i for i in x]\n")
    b = structural_features(b"def f(x):\n  return [i for i in x]\n")
    assert (a == b).all()
    assert len(a) == N_STRUCT_FEATS
    broken = structural_features(b"def f(x:\n  return [i for i in x\n")
    assert broken[1] >= 0  # imbalance slot exists


def test_latency_smoke(arth):
    ids, mask = _ids_mask(b"x = 1\n" * 100)
    joined, spans = mark_options(b"x = 1", [b"a", b"b", b"c", b"d"])
    for _ in range(5):
        p = arth.pooled(ids, mask)
        arth.semantic(p, arth.option_embeds(joined, spans).unsqueeze(0))
    t0 = time.perf_counter()
    for _ in range(10):
        p = arth.pooled(ids, mask)
        arth.semantic(p, arth.option_embeds(joined, spans).unsqueeze(0))
    dt = (time.perf_counter() - t0) / 10 * 1000
    assert dt < 200, f"torch e2e too slow: {dt:.1f}ms"
