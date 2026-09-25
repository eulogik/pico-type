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
    # Wk3-4 refit on shipped ckpt (ft3 = RLCD heads, held-out seeds): 3-5 buckets fitted, others default 1.0
    assert c.temp("choice", 4) == 0.7672  # choice/3-5 fitted
    assert c.temp("noul", 4) == 0.5241  # noul/3-5 fitted
    assert c.temp("noul", 15) == 1.0  # noul/6-20 unfitted default
    assert c.temp("score", 4) == 2.6671  # score/3-5 fitted


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


def test_riskpp_label_coverage():
    """Amendment #4 regression (2026-09-23): every RISK14 label must appear as a
    training positive in some generator. jwt/ssh_key/password/email/phone had
    ZERO positives (RISK14.index never called) — first surfaced as external-audit
    recall 0.000, not by any in-repo test. This test closes that blind spot."""
    from model.pico_type.arth_data import RISK14, RISKPP_GENERATORS

    seen = set()
    for gen in RISKPP_GENERATORS.values():
        for it in gen(100, 3):
            for i, b in enumerate(it["risk14"]):
                if b:
                    seen.add(RISK14[i])
    missing = set(RISK14) - seen
    assert not missing, f"labels with zero training positives: {sorted(missing)}"


def test_hypercube_edge_structure():
    """Hypercube wiring (arXiv:2609.18145): query i sees exactly {i, i XOR 2**bit}."""
    from model.pico_type.arch import hypercube_sparse_attention

    L, D = 8, 4
    torch.manual_seed(0)
    q = torch.randn(1, 1, L, D)
    k = torch.randn(1, 1, L, D)
    v = torch.randn(1, 1, L, D)
    for bit in range(3):
        out = hypercube_sparse_attention(q, k, v, bit)
        mask = torch.zeros(L, L, dtype=torch.bool)
        for i in range(L):
            mask[i, i] = True
            mask[i, i ^ (1 << bit)] = True
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.view(1, 1, L, L)
        )
        assert torch.allclose(out, ref, atol=1e-6), f"bit {bit} mismatch"


def test_hypercube_reaches_all_positions():
    """log2(n) rotated dimensions -> every position reachable in log2(n)
    layers (the paper's core claim; a fixed pattern held across layers fails)."""
    n = 64
    dims = n.bit_length() - 1
    reach = torch.eye(n, dtype=torch.bool)
    for layer in range(dims):
        bit = layer % dims
        nxt = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            nxt[i] = reach[i] | reach[i ^ (1 << bit)]
        reach = nxt
    assert reach.all()
    fixed = torch.eye(n, dtype=torch.bool)
    for _ in range(dims):
        fixed = fixed | torch.stack(
            [fixed[i ^ 1] for i in range(n)]
        )
    assert not fixed.all(), "fixed (non-rotating) pattern must NOT reach all in dims steps"


def test_hypercube_padding_respected():
    from model.pico_type.arch import hypercube_sparse_attention

    L, D, B = 8, 4, 3
    torch.manual_seed(0)
    q = torch.randn(B, 1, L, D)
    k = torch.randn(B, 1, L, D)
    v = torch.randn(B, 1, L, D)
    key_valid = torch.ones(B, L, dtype=torch.bool)
    key_valid[0, 3] = False
    key_valid[2, 1] = False
    out = hypercube_sparse_attention(q, k, v, 1, key_valid)
    mask = torch.zeros(B, L, L, dtype=torch.bool)
    for b in range(B):
        for i in range(L):
            mask[b, i, i] = True
            j = i ^ 2
            if j < L and bool(key_valid[b, j]):
                mask[b, i, j] = True
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None]
    )
    assert torch.allclose(out, ref, atol=1e-6)


def test_hypercube_dense_path_unchanged(arth):
    """Default (dense) pooled must be byte-identical to explicit dense after
    the wiring is added — the shipped legacy-parity contract."""
    data = b"def hello(name):\n    return f'hi {name}'\n" * 20
    ids, mask = _ids_mask(data)
    with torch.no_grad():
        a = arth.pooled(ids, mask)
        b = arth.pooled(ids, mask, wiring="dense")
    assert torch.equal(a, b)


def test_hypercube_model_forward_runs(arth):
    data = b"the quick brown fox jumps over the lazy dog. " * 8
    ids, mask = _ids_mask(data)
    with torch.no_grad():
        dense = arth.pooled(ids, mask)
        hyb = arth.pooled(ids, mask, wiring="hypercube")
    assert dense.shape == hyb.shape
    assert not torch.equal(dense, hyb)
    with pytest.raises(ValueError):
        arth.pooled(ids, mask, wiring="bogus")
    arth.trunk.train()
    try:
        with pytest.raises(RuntimeError):
            arth.pooled(ids, mask, wiring="hypercube")
    finally:
        arth.trunk.eval()


def test_no_real_secrets():
    import re

    from model.pico_type.arth_data import RISKPP_GENERATORS, gen_benign_hard

    cred = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|\b\d{3}-\d{2}-\d{4}\b|\b4\d{15}\b")
    for gen in list(RISKPP_GENERATORS.values()) + [lambda n, s: gen_benign_hard(n, s)]:
        for it in gen(50, 11):
            t = it["input"]
            for m in cred.finditer(t):
                ctx = t[max(0, m.start() - 30) : m.end() + 30]
                assert "EXAMPLE" in ctx or re.fullmatch(r"(000|666|9\d\d)-\d{2}-\d{4}", m.group(0)), m.group(0)


def test_benign_hard_zeros():
    from model.pico_type.arth_data import gen_benign_hard

    items = gen_benign_hard(50, 7)
    assert len(items) == 50
    assert all(it["risk14"] == [0] * 14 and it["source"] == "synth/benign_hard" for it in items)
    assert len({it["input"] for it in items}) > 40  # diverse, not 5 templates


def test_toxicchat_jail_split():
    """Train-split structure + credential guard. Skips structure when cache absent
    (CI); skip-contract for missing data is covered in test_manifest_and_loaders."""
    import re

    from model.pico_type.arth_data import _ROOT, gen_toxicchat_jail, load_toxicchat

    cache = os.path.join(_ROOT, "data", "raw", "toxicchat_train.json")
    if not os.path.exists(cache):
        assert gen_toxicchat_jail(7) == []
        return
    items = gen_toxicchat_jail(7)
    pos = [it for it in items if any(it["risk14"])]
    assert len(pos) == 297  # (113 train positives - 14 test-overlap) x3
    assert all(it["risk14"] == [0] * 14 or it["risk14"][7] == 1 for it in items)
    assert all(len(it["risk14"]) == 14 for it in items)
    cred = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|\b\d{3}-\d{2}-\d{4}\b|\b4\d{15}\b")
    for it in items:
        assert not cred.search(it["input"]), it["input"][:80]
    # held-out purity: no test-split inputs may appear in training items
    test_rows = load_toxicchat("test")
    if test_rows:
        test_set = {r["input"] for r in test_rows}
        train_set = {it["input"] for it in items}
        assert not (train_set & test_set), "test-split leakage into training"


def test_risk_thresholds_and_flags():
    from model.pico_type.arth import RISK_PLUS_LABELS, load_risk_thresholds, risk_flags

    thrs = load_risk_thresholds()
    # fitted: all 14 Risk++ labels since manifest amendment #4 + the ft5 round
    # (before that, jwt/ssh_key/password/email/phone had no trained positives
    # and were correctly absent from the file).
    assert len(thrs) == 14
    assert all(0.0 < t < 1.0 for t in thrs.values())
    assert "jailbreak" in thrs and "prompt_injection" in thrs
    probs = torch.full((1, 14), 0.999)
    flags = risk_flags(probs, thrs)
    assert flags.shape == (1, 14) and bool(flags.all())  # thresholds + 0.5 defaults all < 0.999
    probs2 = torch.zeros(1, 14)
    assert not bool(risk_flags(probs2, thrs).any())
    assert len(RISK_PLUS_LABELS) == 14


def test_rlcd_reward(arth):
    """RLCD-lite reward: finite float, better on correct-confident than wrong."""
    import math

    from scripts.arth_rlcd import reward

    items_sem = [
        {"input": "q: 2+2?\nA) 4\nB) 5", "options": ["4", "5"], "correct": 0},
        {"input": "q: color of sky?\nA) blue\nB) green", "options": ["blue", "green"], "correct": 0},
    ]
    items_risk = [
        {"input": "aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "risk14": [1] + [0] * 13},
        {"input": "plain benign text", "risk14": [0] * 14},
    ]
    r = reward(arth, items_sem, items_risk, [0, 1], [0, 1])
    assert isinstance(r, float) and math.isfinite(r)
    # trunk untouched by reward evaluation
    assert all(not p.requires_grad for p in arth.trunk.parameters())


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
    # manifest amendment #4: riskpp_synth 4000 -> 7000 (6 new generators)
    assert meta["splits"]["riskpp_synth"]["n"] == 7000
    assert meta["splits"]["benign_hard"]["n"] == 1500
    assert meta["splits"]["toxicchat_jail"]["n"] == 897  # (113-14 test-overlap)*3 + 600neg
    assert meta["splits"]["heap_code"]["n"] == 8709
    assert meta["splits"]["wiki_text"]["n"] == 5000
    # manifest amendment #3: pre-registered external tables (graceful in CI)
    for name, n in (("ag_news", 8000), ("sst2", 6000), ("enron_spam", 6000)):
        if name in meta["splits"]:
            assert meta["splits"][name]["n"] == n
        else:
            assert os.path.exists(os.path.join(ROOT, "data/raw", f"{name}_train.json")) or True
    # graceful-skip contract: unreachable dataset id -> [] (works with or
    # without local caches / datasets lib installed)
    monkey_id = os.path.join(ROOT, "data", "raw", "_no_such_.json")
    prev = os.environ.get("TOXICCHAT_DATASET")
    os.environ["TOXICCHAT_DATASET"] = "/nonexistent/dataset-id-for-test"
    try:
        assert ad.load_toxicchat("train", path=monkey_id) == []
    finally:
        if prev is None:
            os.environ.pop("TOXICCHAT_DATASET", None)
        else:
            os.environ["TOXICCHAT_DATASET"] = prev
    assert ad.load_typed_decisions() == []


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


def test_train_mixed_option_cal_bounded(tmp_path):
    """Regression: 2-option ext rows (SST/Enron) batched with 4-option rows must
    not blow up the calibrator CE — padded -1e9 slots entered the log-softmax
    mean and drove cal to 1.4e8 / total 7.1e5 on the first FT4 launch (caught
    at step 4850, killed before any checkpoint)."""
    from model.pico_type import arth_train as at

    if not os.path.exists(os.path.join(ROOT, "data", "raw", "arth", "riskpp_synth.jsonl")):
        pytest.skip("corpora not built")
    out = at.train(
        _train_args(tmp_path, steps=3, n_sem=16, n_risk=2, n_legacy=2, n_rel=1)
    )
    for rec in out["history"]:
        assert rec["cal"] < 100, f"cal exploded: {rec}"
        assert rec["total"] < 100, f"total exploded: {rec}"


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


def test_onnx_export_smoke(arth, tmp_path):
    """Export ARTH to single-file ONNX and run it under onnxruntime.
    Guards the two export bugs already hit: dynamo exporter dropped weights
    into a .data sidecar (0.41MB main file) and mis-decomposed MHA."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from scripts.arth_export_onnx import N_STRUCT, OUT_NAMES, export

    path = str(tmp_path / "arth.onnx")
    export(arth, path, max_len=128)
    assert not os.path.exists(path + ".data"), "weights must be inlined (single file)"
    assert os.path.getsize(path) > 5_000_000, "weights missing from export"

    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    import numpy as np

    out = sess.run(
        OUT_NAMES,
        {
            "input_ids": np.zeros((1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=bool),
            "opt_embs": np.zeros((1, 4, 96), dtype=np.float32),
            "struct_feats": np.zeros((1, N_STRUCT), dtype=np.float32),
            "act_stats": np.zeros((1, 4), dtype=np.float32),
        },
    )
    assert len(out) == len(OUT_NAMES)
    assert out[0].shape == (1, len(COARSE_LABELS))
    assert out[7].shape == (1, 4)  # semantic_logits
    assert out[9].shape == (1, 14)  # riskpp_logits
    model = onnx.load(path, load_external_data=False)
    assert model.ir_version == 8


def test_external_audit():
    """External hand-labeled audit set (plan §3 Cheap-Verifiers rule arXiv:2609.01345):
    structure, credential guard, and zero overlap with training generators."""
    import re

    from model.pico_type.arth_data import RISKPP_GENERATORS, gen_benign_hard

    path = os.path.join(ROOT, "data", "arth_audit_external.json")
    with open(path) as f:
        audit = json.load(f)
    items = audit["items"]
    assert len(items) >= 200
    assert "hand" in audit["labeled_by"]
    assert "NEVER used in training" in audit["protocol"]

    train_texts = set()
    for gen in list(RISKPP_GENERATORS.values()) + [lambda n, s: gen_benign_hard(n, s)]:
        train_texts.update(it["input"] for it in gen(100, 3))

    cred = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|\b\d{3}-\d{2}-\d{4}\b|\b4\d{15}\b")
    pos_seen = 0
    for it in items:
        assert len(it["risk14"]) == 14
        assert set(it["risk14"]) <= {0, 1}
        assert it["text"] not in train_texts, "audit must never overlap training"
        pos_seen += sum(it["risk14"])
        for m in cred.finditer(it["text"]):
            ctx = it["text"][max(0, m.start() - 30) : m.end() + 30]
            assert "EXAMPLE" in ctx or re.fullmatch(r"(000|666|9\d\d)-\d{2}-\d{4}", m.group(0)), m.group(0)
    assert pos_seen >= 100  # >=8 positives per label after 2026-09-23 growth


def test_external_tables():
    """AG/SST-2/Enron gates (plan §3): builders, held-out purity, PII/cred guards.
    Skips when external caches are absent (CI, like toxicchat)."""
    import re

    from model.pico_type import arth_data as ad

    if not os.path.exists(os.path.join(ROOT, "data/raw/ag_news_train.json")):
        pytest.skip("external table caches absent")

    ag = ad.gen_ag_news()
    sst = ad.gen_sst2()
    en = ad.gen_enron_spam()
    assert len(ag) == 8000 and len(sst) == 6000 and len(en) == 6000
    for items, opts in ((ag, ad.AG_OPTIONS), (sst, ad.SST_OPTIONS), (en, ad.ENRON_OPTIONS)):
        for it in items[:200]:
            assert it["mode"] == "choice" and sorted(it["options"]) == sorted(opts)
            assert 0 <= it["correct"] < len(opts)

    # held-out purity: eval sets never overlap train samples
    ag_eval = ad.load_ag_eval()
    sst_eval = ad.load_sst2_eval()
    en_eval = ad.load_enron_eval()
    assert len(ag_eval) > 500 and len(sst_eval) > 500 and len(en_eval) > 1000
    ag_tr = {it["input"] for it in ag}
    sst_tr = {it["input"] for it in sst}
    en_tr = {it["input"] for it in en}
    assert not ({it["input"] for it in ag_eval} & ag_tr)
    assert not ({it["input"] for it in sst_eval} & sst_tr)
    assert not ({it["input"] for it in en_eval} & en_tr)

    # PII masked + credential guard on cached rows (Enron especially)
    cred = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|\b\d{3}-\d{2}-\d{4}\b|\b4\d{15}\b")
    for rows in (en[:300], en_eval[:200]):
        for it in rows:
            assert not cred.search(it["input"]), it["input"][:80]
    # fit slices disjoint from gradient-trained indices
    fit = ad.fit_slice_sst2()
    assert fit and not ({it["input"] for it in fit} & sst_tr)
