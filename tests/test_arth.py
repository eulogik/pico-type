"""ARTH Wk 1-2 tests: head shapes, frozen-trunk parity, calibrator, latency smoke."""

from __future__ import annotations

import argparse
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


def test_riskpp_generators():
    from model.pico_type.arth_data import RISK14, RISKPP_GENERATORS

    assert len(RISK14) == 14
    total_pos = 0
    for gen in RISKPP_GENERATORS.values():
        items = gen(20, 7)
        assert len(items) == 20
        for it in items:
            assert len(it["risk14"]) == 14
            assert set(it["risk14"]) <= {0, 1}
            assert len(it["input"]) > 0
        total_pos += sum(1 for it in items if any(it["risk14"]))
    assert total_pos > len(RISKPP_GENERATORS) * 10


def test_no_real_secrets():
    import re

    from model.pico_type.arth_data import RISKPP_GENERATORS

    cred = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|\b\d{3}-\d{2}-\d{4}\b|\b4\d{15}\b")
    for gen in RISKPP_GENERATORS.values():
        for it in gen(50, 11):
            t = it["input"]
            for m in cred.finditer(t):
                ctx = t[max(0, m.start() - 30) : m.end() + 30]
                assert "EXAMPLE" in ctx or re.fullmatch(r"(000|666|9\d\d)-\d{2}-\d{4}", m.group(0)), m.group(0)


def test_decision_shapes():
    from model.pico_type import arth_data as ad

    ch = ad.make_choice_decisions(6, 7)
    assert len(ch) == 6
    for it in ch:
        assert len(it["options"]) == 4 and 0 <= it["correct"] < 4 and it["mode"] == "choice"
    sc = ad.make_score_decisions(4, 7)
    for it in sc:
        assert len(it["scores"]) == len(it["options"]) and it["mode"] == "score"
    nl = ad.make_noul_decisions(4, 7)
    assert all(it["correct"] is None and it["mode"] == "noul" for it in nl)
    st = ad.make_structural_pairs(4, 7)
    assert all(len(it["options"]) == 4 and 0 <= it["correct"] < 4 for it in st)
    un = ad.make_unanswerable(4, 7)
    assert all(it["correct"] is None and "TOTAL:" not in it["input"] for it in un)


def test_data_determinism():
    from model.pico_type import arth_data as ad

    a = ad.gen_xss_payload(10, 7)
    b = ad.gen_xss_payload(10, 7)
    assert [ad.item_hash(x) for x in a] == [ad.item_hash(x) for x in b]
    assert ad.item_hash(a[0]) == ad.item_hash(dict(a[0]))


def test_manifest_and_loaders():
    import json

    from model.pico_type import arth_data as ad

    with open(os.path.join(ROOT, "data", "arth_manifest.json")) as f:
        meta = json.load(f)
    assert meta["frozen"] is True and meta["seed"] == 7
    assert meta["splits"]["riskpp_synth"]["n"] == 4000
    assert meta["splits"]["heap_code"]["n"] == 8709
    assert meta["splits"]["wiki_text"]["n"] == 5000
    assert ad.load_enron() == [] and ad.load_toxicchat() == [] and ad.load_typed_decisions() == []


def test_pooled_grad_flow(arth):
    """Phase A: frozen trunk -> pooled has no grad. Phase B: unfrozen -> grad flows
    (regression: pooled() used to hard-wrap no_grad, starving trunk of sem/risk/rel grads)."""
    ids, mask = _ids_mask(b"def f(x): return x")
    p = arth.pooled(ids, mask)
    assert not p.requires_grad  # Phase A (fixture trunk frozen)
    for prm in arth.trunk.parameters():
        prm.requires_grad_(True)
    try:
        p2 = arth.pooled(ids, mask)
        assert p2.requires_grad  # Phase B: trunk MUST see grads from pooled paths
    finally:
        arth.trunk.requires_grad_(False)
        arth.trunk.eval()


def test_latest_ckpt_numeric_sort(tmp_path):
    from model.pico_type.arth_train import latest_ckpt

    for n in (50, 100, 900):
        (tmp_path / f"arth_step_{n}.pt").write_bytes(b"x")
    assert latest_ckpt(str(tmp_path)).endswith("arth_step_900.pt")  # not lexicographic 900<100
    (tmp_path / "arth_step_1000.pt").write_bytes(b"x")
    assert latest_ckpt(str(tmp_path)).endswith("arth_step_1000.pt")
    assert latest_ckpt(str(tmp_path / "missing")) is None


def test_train_losses_finite():
    from model.pico_type.arth_train import Streams, encode

    if not os.path.exists(os.path.join(ROOT, "data", "raw", "arth", "riskpp_synth.jsonl")):
        pytest.skip("corpora not built")
    st = Streams(seed=7, subset=10)
    leg, sem, rsk, rel = st.sample(2, 2, 2, 1)
    assert len(leg) == 2 and len(sem) == 2 and len(rsk) == 2 and len(rel) == 1
    ids, mask = encode([b"hello world", b"\x89PNG\r\n"])
    assert ids.shape[0] == 2 and mask.shape[0] == 2
    assert int(mask[0].sum()) == 11


def _train_args(tmp_path, **kw):
    base = {
        "steps": 2, "subset": 20, "freeze_steps": 2000, "n_legacy": 2, "n_sem": 2,
        "n_risk": 2, "n_rel": 1, "lr": 3e-4, "lr_trunk": 3e-5, "kd_temp": 2.0,
        "w_legacy": 1.0, "w_sem": 1.0, "w_rel": 0.5, "w_risk": 0.5,
        "lambda_cal": 0.005, "out": str(tmp_path), "log_every": 1,
        "save_every": 2, "resume": False, "seed": 7,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_train_two_steps(tmp_path):
    from model.pico_type import arth_train as at

    if not os.path.exists(os.path.join(ROOT, "data", "raw", "arth", "riskpp_synth.jsonl")):
        pytest.skip("corpora not built")
    out = at.train(_train_args(tmp_path))
    import math

    assert len(out["history"]) == 2
    for rec in out["history"]:
        assert all(abs(v) != float("inf") and not math.isnan(v) for k, v in rec.items() if k != "step")
    assert os.path.exists(os.path.join(str(tmp_path), "arth_final.pt"))


def test_resume_across_unfreeze(tmp_path):
    """Regression: post-unfreeze ckpt (2 param_groups) must resume without
    ValueError and keep training (was: load_state_dict group-count crash)."""
    from model.pico_type import arth_train as at

    if not os.path.exists(os.path.join(ROOT, "data", "raw", "arth", "riskpp_synth.jsonl")):
        pytest.skip("corpora not built")
    # phase A step0 (1-group save), unfreeze at step1, step1 save (2-group)
    at.train(_train_args(tmp_path, steps=2, freeze_steps=1, save_every=1))
    assert at.latest_ckpt(str(tmp_path)).endswith("arth_step_2.pt")
    out = at.train(_train_args(tmp_path, steps=3, freeze_steps=1, save_every=1, resume=True))
    assert len(out["history"]) == 1  # only step 2 left
    assert out["history"][0]["step"] == 2
    import math

    assert all(
        math.isfinite(v) for k, v in out["history"][0].items() if k != "step"
    )


def test_verify_manifest():
    from model.pico_type.arth_data import verify_manifest

    assert verify_manifest() is True


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
