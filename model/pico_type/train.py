"""pico-type: multi-task trainer."""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .arch import PicoType, PicoTypeConfig, encode_bytes
from .data import IGNORE_INDEX, MAX_BYTES, MIN_BYTES, Sample, SyntheticGenerator, SyntheticDataset
from .labels import HEAD_NUM_CLASSES

ALL_HEADS = ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk")
SINGLE_LABEL_HEADS = ALL_HEADS[:-1]

DEFAULT_HEAD_WEIGHTS: Dict[str, float] = {
    "coarse": 3.0,
    "modality": 2.0,
    "subtype": 1.0,
    "code_lang": 1.5,
    "text_lang": 1.5,
    "file_mime": 1.0,
    "risk": 1.0,
}


@dataclass
class TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 100
    total_steps: int = 5000
    batch_size: int = 64
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 500
    eval_every: int = 500
    train_size: int = 10000
    eval_size: int = 500
    output_dir: str = "checkpoints"
    model_config: PicoTypeConfig = field(default_factory=PicoTypeConfig)
    head_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAD_WEIGHTS))
    seed: int = 42
    device: str = "auto"
    compile: bool = False
    tier: str = "base"
    train_tiers: Tuple[str, ...] = ("tiny", "small", "base", "pro")


def collate_fn(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    max_len = max(len(s.data) for s in batch)
    max_len = min(max_len, MAX_BYTES)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels: Dict[str, torch.Tensor] = {}
    for head in SINGLE_LABEL_HEADS:
        labels[head] = torch.full((len(batch),), IGNORE_INDEX, dtype=torch.long)
    risk_labels = torch.zeros(len(batch), HEAD_NUM_CLASSES["risk"], dtype=torch.float)

    for i, s in enumerate(batch):
        data = s.data[:max_len]
        input_ids[i, : len(data)] = torch.tensor(list(data), dtype=torch.long)
        attention_mask[i, : len(data)] = 1
        for head in SINGLE_LABEL_HEADS:
            v = getattr(s, head)
            if v != IGNORE_INDEX:
                labels[head][i] = v
        for r in s.risk:
            risk_labels[i, r] = 1.0
    labels["risk"] = risk_labels
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask.bool() & attention_mask.bool(),
        "labels": labels,
    }


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=IGNORE_INDEX)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, torch.Tensor] = {}
        for head in SINGLE_LABEL_HEADS:
            lbl = labels[head]
            if (lbl != IGNORE_INDEX).sum() > 0:
                losses[head] = self.ce(logits[head], lbl)
            else:
                losses[head] = torch.tensor(0.0, device=lbl.device)
        losses["risk"] = self.bce(logits["risk"], labels["risk"])

        total = torch.zeros(1, device=next(iter(logits.values())).device)
        individual: Dict[str, float] = {}
        for head, loss in losses.items():
            w = self.weights.get(head, 1.0)
            total = total + w * loss
            individual[head] = loss.detach().item()
        individual["total"] = total.detach().item()
        return total, individual


def multi_tier_loss(
    model: PicoType,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    criterion: MultiTaskLoss,
    tiers: Iterable[str],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    tier_losses: List[torch.Tensor] = []
    summed_components: Dict[str, float] = {}
    count = 0
    for tier in tiers:
        logits = model(input_ids, attention_mask, tier=tier)
        loss, components = criterion(logits, labels)
        tier_losses.append(loss)
        count += 1
        for key, value in components.items():
            summed_components[key] = summed_components.get(key, 0.0) + value
            summed_components[f"{tier}/{key}"] = value
    if not tier_losses:
        raise ValueError("at least one training tier is required")
    total = torch.stack([loss.reshape(()) for loss in tier_losses]).mean()
    averaged = {
        key: value / count
        for key, value in summed_components.items()
        if "/" not in key
    }
    averaged["total"] = total.detach().item()
    return total, {**averaged, **{k: v for k, v in summed_components.items() if "/" in k}}


def get_lr(it: int, config: TrainConfig) -> float:
    if it < config.warmup_steps:
        return config.lr * (it + 1) / config.warmup_steps
    progress = (it - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: Optional[TrainConfig] = None) -> TrainConfig:
    config = config or TrainConfig()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    gen = SyntheticGenerator(seed=config.seed)
    train_ds = SyntheticDataset(gen, config.train_size)
    eval_ds = SyntheticDataset(SyntheticGenerator(seed=config.seed + 1), config.eval_size)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=0)

    device = torch.device(config.device)
    model = PicoType(config.model_config).to(device)
    criterion = MultiTaskLoss(config.head_weights)

    param_groups: list = [
        {"params": [p for n, p in model.named_parameters() if "linears" not in n], "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if "linears" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=config.betas)

    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    os.makedirs(config.output_dir, exist_ok=True)
    step = 0
    best_loss = float("inf")
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu") if device.type == "cuda" else None
    amp_dtype = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    amp_ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    with open(os.path.join(config.output_dir, "train_config.json"), "w") as f:
        json.dump(
            {
                "lr": config.lr,
                "total_steps": config.total_steps,
                "batch_size": config.batch_size,
                "train_tiers": list(config.train_tiers),
            },
            f,
        )

    while step < config.total_steps:
        model.train()
        for batch in train_loader:
            if step >= config.total_steps:
                break

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            optimizer.zero_grad()
            with amp_ctx:
                loss, loss_components = multi_tier_loss(
                    model,
                    input_ids,
                    attention_mask,
                    labels,
                    criterion,
                    config.train_tiers,
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            if step % config.log_every == 0:
                parts = [f"step={step:5d} lr={lr:.6f} loss={loss.item():.4f}"]
                for h, v in loss_components.items():
                    if "/" in h:
                        continue
                    parts.append(f"{h}={v:.4f}")
                print("  ".join(parts))

            if step % config.save_every == 0 and step > 0:
                ckpt = {"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss.item()}
                torch.save(ckpt, os.path.join(config.output_dir, f"step_{step}.pt"))

            if step % config.eval_every == 0:
                model.eval()
                eval_losses: Dict[str, float] = {"total": 0.0}
                eval_batches = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        ids = eval_batch["input_ids"].to(device)
                        mask = eval_batch["attention_mask"].to(device)
                        lbls = {k: v.to(device) for k, v in eval_batch["labels"].items()}
                        with amp_ctx:
                            _, comps = multi_tier_loss(
                                model,
                                ids,
                                mask,
                                lbls,
                                criterion,
                                config.train_tiers,
                            )
                        for k, v in comps.items():
                            if "/" in k:
                                continue
                            eval_losses[k] = eval_losses.get(k, 0.0) + v
                        eval_batches += 1
                n = max(1, eval_batches)
                parts = [f" eval step={step:5d}"]
                averaged_eval_losses = {}
                for k, v in eval_losses.items():
                    averaged_eval_losses[k] = v / n
                    parts.append(f"{k}={averaged_eval_losses[k]:.4f}")
                print("  ".join(parts))
                if averaged_eval_losses.get("total", float("inf")) < best_loss:
                    best_loss = averaged_eval_losses["total"]
                    ckpt = {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval_loss": best_loss,
                    }
                    torch.save(ckpt, os.path.join(config.output_dir, "best.pt"))
                model.train()

            step += 1

    final = {"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "final_loss": loss.item()}
    torch.save(final, os.path.join(config.output_dir, "final.pt"))
    return config


def load_checkpoint(path: str, model: PicoType, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[PicoType, Optional[torch.optim.Optimizer], Dict]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict")}
