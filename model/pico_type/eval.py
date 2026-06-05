"""pico-type: eval harness."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .arch import PicoType, PicoTypeConfig


def _average_precision(y_true, y_scores):
    """Average precision as precision-at-positive averaged over positives."""
    import numpy as np
    order = y_scores.argsort()[::-1]
    y_true = y_true[order]
    positives = y_true.sum()
    if positives <= 0:
        return 0.0
    tp = y_true.cumsum()
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    precision = tp / ranks
    return float((precision * y_true).sum() / positives)
from .data import (
    IGNORE_INDEX,
    SyntheticGenerator,
    SyntheticDataset,
    Sample,
)
from .labels import (
    COARSE_LABELS,
    MODALITY_LABELS,
    SUBTYPE_LABELS,
    CODE_LANG_LABELS,
    TEXT_LANG_LABELS,
    FILE_MIME_LABELS,
    RISK_LABELS,
    HEAD_NUM_CLASSES,
    label_for,
)
from .train import collate_fn

ALL_HEADS = ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk")
SINGLE_LABEL_HEADS = ALL_HEADS[:-1]


@dataclass
class EvalConfig:
    checkpoint: str = ""
    model_config: PicoTypeConfig = field(default_factory=PicoTypeConfig)
    tier: str = "base"
    eval_size: int = 1000
    batch_size: int = 64
    seed: int = 42
    device: str = "auto"
    result_path: str = ""


@dataclass
class HeadMetrics:
    accuracy: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    support: int = 0
    n_classes: int = 0
    confusion: List[List[int]] = field(default_factory=list)


@dataclass
class RiskMetrics:
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    average_precision: float = 0.0


@dataclass
class EvalResults:
    head_metrics: Dict[str, HeadMetrics] = field(default_factory=dict)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    inference_time_ms: float = 0.0
    config: Dict = field(default_factory=dict)


def _per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
    n_classes: int,
) -> Tuple[Dict[str, Dict[str, float]], List[List[int]]]:
    conf = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(labels):
        tp = conf[i][i]
        fp = sum(conf[j][i] for j in range(n_classes)) - tp
        fn = sum(conf[i][j] for j in range(n_classes)) - tp
        eps = 1e-10
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        metrics[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(sum(conf[i])),
        }
    return metrics, conf


def evaluate(config: EvalConfig) -> EvalResults:
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(config.device)

    gen = SyntheticGenerator(seed=config.seed)
    ds = SyntheticDataset(gen, config.eval_size)
    loader = DataLoader(ds, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=0)

    model = PicoType(config.model_config).to(device)
    if config.checkpoint and os.path.exists(config.checkpoint):
        from .train import load_checkpoint
        load_checkpoint(config.checkpoint, model)
    elif config.checkpoint:
        raise FileNotFoundError(f"checkpoint not found: {config.checkpoint}")
    model.eval()

    results: Dict[str, List[int]] = {head: [] for head in SINGLE_LABEL_HEADS}
    results["risk_pred"] = []
    results["risk_true"] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(ids, mask, tier=config.tier)

            for head in SINGLE_LABEL_HEADS:
                true_vals = labels[head]
                pred_vals = logits[head].argmax(dim=-1).cpu()
                for t, p in zip(true_vals, pred_vals):
                    if t.item() != IGNORE_INDEX:
                        results[head].append((t.item(), p.item()))

            risk_true_batch = labels["risk"].cpu()
            risk_pred_batch = torch.sigmoid(logits["risk"]).cpu()
            for t, p in zip(risk_true_batch, risk_pred_batch):
                results["risk_true"].append(t.numpy())
                results["risk_pred"].append(p.numpy())
    elapsed = time.perf_counter() - t0

    time_ms = elapsed / len(ds) * 1000.0
    head_metrics: Dict[str, HeadMetrics] = {}
    for head in SINGLE_LABEL_HEADS:
        items = results[head]
        if not items:
            head_metrics[head] = HeadMetrics()
            continue
        y_true, y_pred = zip(*items)
        n_classes = HEAD_NUM_CLASSES[head]
        labels_list = {
            "coarse": COARSE_LABELS,
            "modality": MODALITY_LABELS,
            "subtype": SUBTYPE_LABELS,
            "code_lang": CODE_LANG_LABELS,
            "text_lang": TEXT_LANG_LABELS,
            "file_mime": FILE_MIME_LABELS,
        }[head]
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
        pclass, conf = _per_class_metrics(list(y_true), list(y_pred), labels_list, n_classes)
        head_metrics[head] = HeadMetrics(
            accuracy=round(acc, 4),
            per_class=pclass,
            support=len(y_true),
            n_classes=n_classes,
            confusion=conf,
        )

    risk_items_true = results["risk_true"]
    risk_items_pred = results["risk_pred"]
    risk_pclass: Dict[str, Dict[str, float]] = {}
    if risk_items_true:
        import numpy as np
        risk_arr = torch.from_numpy(np.asarray(risk_items_true))
        pred_arr = torch.from_numpy(np.asarray(risk_items_pred))
        ap_values = []
        for i in range(len(RISK_LABELS)):
            y_t = risk_arr[:, i].numpy()
            y_p = pred_arr[:, i].numpy()
            ap = float(_average_precision(y_t, y_p))
            ap_values.append(ap)
            risk_pclass[RISK_LABELS[i]] = {"average_precision": round(ap, 4)}
    risk_metrics = RiskMetrics(
        per_class=risk_pclass,
        average_precision=round(sum(ap_values) / max(1, len(ap_values)), 4) if risk_items_true else 0.0,
    )

    return EvalResults(
        head_metrics=head_metrics,
        risk_metrics=risk_metrics,
        inference_time_ms=round(time_ms, 4),
        config={
            "checkpoint": config.checkpoint,
            "tier": config.tier,
            "eval_size": config.eval_size,
            "batch_size": config.batch_size,
            "model_max_bytes": config.model_config.max_bytes,
        },
    )


def run_eval(args: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="pico-type eval harness")
    p.add_argument("--checkpoint", "-c", default="", help="model checkpoint path")
    p.add_argument("--eval-size", type=int, default=1000, help="number of samples")
    p.add_argument("--batch-size", type=int, default=64, help="batch size")
    p.add_argument("--tier", default="base", choices=["tiny", "small", "base", "pro"])
    p.add_argument("--output", "-o", default="", help="output JSON path")
    p.add_argument("--seed", type=int, default=42)
    parsed = p.parse_args(args)

    cfg = EvalConfig(
        checkpoint=parsed.checkpoint,
        eval_size=parsed.eval_size,
        batch_size=parsed.batch_size,
        tier=parsed.tier,
        seed=parsed.seed,
        result_path=parsed.output,
    )
    results = evaluate(cfg)
    result_dict = _serialize(results)
    print(json.dumps(result_dict, indent=2))
    if parsed.output:
        with open(parsed.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved to {parsed.output}")


def _serialize(results: EvalResults) -> Dict:
    def _hm(hm: HeadMetrics):
        return {
            "accuracy": hm.accuracy,
            "support": hm.support,
            "n_classes": hm.n_classes,
            "per_class": hm.per_class,
            "confusion": hm.confusion,
        }
    return {
        "head_metrics": {k: _hm(v) if isinstance(v, HeadMetrics) else v for k, v in results.head_metrics.items()},
        "risk_metrics": results.risk_metrics.per_class,
        "risk_average_precision": results.risk_metrics.average_precision,
        "inference_time_ms": results.inference_time_ms,
        "config": results.config,
    }


if __name__ == "__main__":
    run_eval()
