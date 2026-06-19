"""Continue training with real code samples from GitHub."""

from __future__ import annotations

import time
from typing import List

import os

import torch
from torch.utils.data import DataLoader

from .arch import PicoType, PicoTypeConfig
from .data import SyntheticGenerator, SyntheticDataset
from .labels import CODE_LANG_LABELS
from .train import TrainConfig, collate_fn, get_lr, multi_tier_loss, MultiTaskLoss
from .realdata import build_real_code_dataset


def train_with_real_data(
    config: TrainConfig,
    token: str = "",
    samples_per_lang: int = 20,
    synthetic_ratio: float = 0.3,
) -> TrainConfig:
    """Train with a mix of real GitHub code and synthetic data."""

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(config.device)

    print("=== Fetching real code samples from GitHub ===")
    real_samples, stats = build_real_code_dataset(
        languages=CODE_LANG_LABELS,
        samples_per_lang=samples_per_lang,
        token=token,
    )
    total_real = sum(stats.values())
    print(f"\nTotal real code samples: {total_real}")
    for lang, count in stats.items():
        if count == 0:
            print(f"  WARNING: No samples for {lang}")

    total_per_epoch = max(config.train_size, total_real)
    real_per_epoch = len(real_samples)
    synth_per_epoch = total_per_epoch - real_per_epoch

    if synth_per_epoch < 0:
        synth_per_epoch = 0
        real_used = real_samples[:config.train_size]
    else:
        real_used = real_samples

    print(f"\nPer epoch: {len(real_used)} real + {synth_per_epoch} synthetic = {total_per_epoch} total")

    gen = SyntheticGenerator(seed=config.seed)
    eval_gen = SyntheticGenerator(seed=config.seed + 1)
    train_synth = SyntheticDataset(gen, synth_per_epoch)
    eval_ds = SyntheticDataset(eval_gen, config.eval_size)

    train_loader = DataLoader(
        train_synth, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=config.batch_size,
        collate_fn=collate_fn, num_workers=0,
    )

    model = PicoType(config.model_config).to(device)
    criterion = MultiTaskLoss(config.head_weights)

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "linears" not in n], "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if "linears" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=config.betas)

    if config.resume_from:
        from .train import load_checkpoint
        path = config.resume_from
        print(f"Resuming from {path}")
        _, _, meta = load_checkpoint(path, model, optimizer)
        resumed_step = meta.get("step", 0)
        best_loss = meta.get("eval_loss", meta.get("loss", float("inf")))
        print(f"  Resumed at step {resumed_step}, best_loss={best_loss:.4f}")
    else:
        resumed_step = 0
        best_loss = float("inf")

    os.makedirs(config.output_dir, exist_ok=True)

    step = 0
    target_steps = config.total_steps
    print(f"\n=== Training {target_steps} additional steps (resumed from {resumed_step}) ===")
    while step < target_steps:
        model.train()
        for batch in train_loader:
            if step >= config.total_steps:
                break

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Mix in real code samples
            real_batch_size = max(1, int(config.batch_size * 0.3))
            real_batch = real_samples[step % len(real_used):][:real_batch_size] if len(real_used) > 0 else []
            if len(real_batch) < real_batch_size:
                real_batch = real_used[:real_batch_size] if len(real_used) > 0 else []

            if real_batch:
                real_collated = collate_fn(real_batch)
                for k, v in real_collated.items():
                    if isinstance(v, dict):
                        real_collated[k] = {hk: hv.to(device) for hk, hv in v.items()}
                    else:
                        real_collated[k] = v.to(device)

            synthetic_collated = batch
            input_ids = synthetic_collated["input_ids"].to(device)
            attention_mask = synthetic_collated["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            optimizer.zero_grad()
            loss, loss_components = multi_tier_loss(
                model, input_ids, attention_mask, labels, criterion, config.train_tiers,
            )

            # Add real code loss
            if real_batch:
                real_ids = real_collated["input_ids"].to(device)
                real_mask = real_collated["attention_mask"].to(device)
                real_labels = {k: v.to(device) for k, v in real_collated["labels"].items()}
                real_loss, real_comps = multi_tier_loss(
                    model, real_ids, real_mask, real_labels, criterion, config.train_tiers,
                )
                loss = loss + real_loss
                for k, v in real_comps.items():
                    loss_components[f"real_{k}"] = v

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            if step % config.log_every == 0:
                parts = [f"step={step:5d} lr={lr:.6f} loss={loss.item():.4f}"]
                for h in ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk"):
                    if h in loss_components:
                        parts.append(f"{h}={loss_components[h]:.4f}")
                print("  ".join(parts))

            step += 1

    final_path = os.path.join(config.output_dir, "real_trained_final.pt")
    torch.save({"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, final_path)
    print(f"\nSaved to {final_path}")

    # Quick eval
    model.eval()
    eval_loss_total = 0.0
    eval_batches = 0
    with torch.no_grad():
        for eval_batch in eval_loader:
            ids = eval_batch["input_ids"].to(device)
            mask = eval_batch["attention_mask"].to(device)
            lbls = {k: v.to(device) for k, v in eval_batch["labels"].items()}
            _, comps = multi_tier_loss(model, ids, mask, lbls, criterion, config.train_tiers)
            eval_loss_total += comps.get("total", 0)
            eval_batches += 1

    avg = eval_loss_total / max(1, eval_batches)
    print(f"Eval loss after real-data training: {avg:.4f}")
    if avg < best_loss:
        torch.save({"step": step, "model_state_dict": model.state_dict(), "eval_loss": avg}, os.path.join(config.output_dir, "best.pt"))
        print("New best.pt saved!")

    return config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="", help="GitHub token")
    parser.add_argument("--samples-per-lang", type=int, default=20)
    parser.add_argument("--total-steps", type=int, default=2000, help="Number of ADDITIONAL steps beyond resume")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", default="checkpoints/best.pt", help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = TrainConfig(
        lr=args.lr,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        warmup_steps=100,
        train_size=10000,
        eval_size=500,
        log_every=25,
        output_dir="checkpoints",
        train_tiers=("base",),
        tier="base",
        resume_from=args.resume,
    )
    train_with_real_data(cfg, token=args.token, samples_per_lang=args.samples_per_lang)
