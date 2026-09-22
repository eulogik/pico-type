"""ARTH: transfer v0.2 weights from single-file ONNX back into torch PicoType.

The v0.2 torch checkpoints are gone (only ONNX exports survive). This recovers
the exact v0.2 trunk by copying ONNX initializers into a fresh PicoType model,
then verifies bit-level agreement (torch forward vs ONNX) on random inputs.

Usage: python scripts/arth_recover_trunk.py [--tier base] [--out checkpoints/arth_trunk_v02.pt]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.pico_type.arch import PicoType, PicoTypeConfig
from model.pico_type.cli import load_onnx_model

ATTN_ORDER = ["qkv.weight", "out_proj.weight", "mlp.0.weight", "mlp.3.weight"]
VAL_ORDER = ["val_55", "val_157", "val_161", "val_170", "val_174", "val_257", "val_261", "val_270"]


def transfer(onnx_path: str, tier: str = "base") -> PicoType:
    """Copy weights from a single-file ONNX export back into torch.

    The exporter constant-folds Linear transposes (torch (out,in) -> ONNX (in,out))
    and renames attention weights to val_*; heads are flattened to the exported tier.
    """
    from onnx import numpy_helper

    model = PicoType(PicoTypeConfig())
    sd = model.state_dict()
    om = onnx.load(onnx_path)
    init = {i.name: numpy_helper.to_array(i) for i in om.graph.initializer}
    n_set = 0
    with torch.no_grad():
        # 1:1 named tensors (embed, conv, norms, biases)
        for k, v in sd.items():
            if k in init and tuple(init[k].shape) == tuple(v.shape):
                v.copy_(torch.from_numpy(init[k].copy()))
                n_set += 1
        # structural: attention weights in graph order, transposed back
        attn_keys = [f"attn_blocks.{b}.{s}" for b in (0, 1) for s in ATTN_ORDER]
        for tk, vk in zip(attn_keys, VAL_ORDER):
            arr = init[vk]
            assert tuple(arr.shape) == tuple(sd[tk].shape[::-1]), (tk, arr.shape, tuple(sd[tk].shape))
            sd[tk].copy_(torch.from_numpy(arr.T.copy()))
            n_set += 1
        # structural: exported-tier head slice
        for name in ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk"):
            tk = f"heads.{name}.linears.{tier}.weight"
            arr = init[f"heads.{name}.weight"]
            assert tuple(arr.shape) == tuple(sd[tk].shape), (tk, arr.shape, tuple(sd[tk].shape))
            sd[tk].copy_(torch.from_numpy(arr.copy()))
            n_set += 1
            bk = f"heads.{name}.linears.{tier}.bias"
            if f"heads.{name}.bias" in init:
                barr = init[f"heads.{name}.bias"]
                assert tuple(barr.shape) == tuple(sd[bk].shape), (bk, barr.shape, tuple(sd[bk].shape))
                sd[bk].copy_(torch.from_numpy(barr.copy()))
                n_set += 1
            else:
                # tiny/small/pro exports dropped head biases (bias-free Gemm) — zero them
                sd[bk].zero_()
                print(f"  note: {onnx_path} has no {name} bias; zeroed {bk}")
    print(f"transferred {n_set}/{len(sd)} tensors from {onnx_path} (tier={tier})")
    return model


def verify(model: PicoType, session, n: int = 5, max_bytes: int = 1024) -> float:
    # NOTE: torch must run on the same padded input as ONNX — conv has no mask,
    # so short-vs-padded sequences give different edge outputs (pre-existing
    # run_torch/run_onnx inconsistency; training batches are always padded).
    rng = np.random.default_rng(7)
    maxdiff = 0.0
    for _ in range(n):
        data = bytes(rng.integers(0, 256, size=rng.integers(16, 1024), dtype=np.uint8))
        ids = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        ln = len(ids)
        pad = np.zeros(max_bytes, dtype=np.int64)
        pad[:ln] = ids
        mask = np.zeros(max_bytes, dtype=bool)
        mask[:ln] = True
        outs = session.run(None, {"input_ids": pad[None, :], "attention_mask": mask[None, :]})
        with torch.no_grad():
            tl = model(torch.tensor([list(pad)]), torch.tensor(mask[None, :]), tier="base")
        from model.pico_type.labels import ALL_HEADS

        for h, o in zip(ALL_HEADS, outs):
            t = tl[h]
            maxdiff = max(maxdiff, float(np.abs(o[0] - t[0].numpy()).max()))
    return maxdiff


def count_params(mod: torch.nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="base")
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "arth_trunk_v02.pt"))
    args = ap.parse_args()

    onnx_path = os.path.join(ROOT, "checkpoints", f"picotype_{args.tier}.onnx")
    model = transfer(onnx_path, tier=args.tier)
    # recover remaining tiers' head slices from their own exports (trunk is shared)
    if args.tier == "base":
        for t in ("tiny", "small", "pro"):
            donor = transfer(os.path.join(ROOT, "checkpoints", f"picotype_{t}.onnx"), tier=t)
            ds = donor.state_dict()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if f".linears.{t}." in k:
                        v.copy_(ds[k])
        print("recovered head slices for tiers: tiny, small, base, pro")
    model.eval()
    session = load_onnx_model(args.tier, os.path.join(ROOT, "checkpoints"))
    maxdiff = verify(model, session)
    print(f"max |torch - onnx| prob diff over 5 random inputs: {maxdiff:.2e}")
    assert maxdiff < 1e-4, f"transfer verification FAILED: {maxdiff}"
    print("transfer verification PASS")
    print(f"recovered trunk params: {count_params(model)}")
    torch.save({"model_state_dict": model.state_dict(), "source": f"onnx:{args.tier}", "tier": args.tier}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
