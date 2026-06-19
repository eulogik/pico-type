"""End-to-end pipeline: fetch real code → fine-tune → evaluate → export."""

from __future__ import annotations

import argparse
import os
import pickle

import torch

from .arch import PicoType, PicoTypeConfig
from .data import SyntheticGenerator, SyntheticDataset
from .labels import CODE_LANG_LABELS
from .train import TrainConfig, collate_fn, get_lr, multi_tier_loss, MultiTaskLoss
from .labels import COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS, TEXT_LANG_LABELS, FILE_MIME_LABELS, RISK_LABELS

REAL_WORLD_LABEL_MAP = {
    "coarse": COARSE_LABELS,
    "modality": MODALITY_LABELS,
    "subtype": SUBTYPE_LABELS,
    "code_lang": CODE_LANG_LABELS,
    "text_lang": TEXT_LANG_LABELS,
    "file_mime": FILE_MIME_LABELS,
    "risk": RISK_LABELS,
}

REAL_WORLD_INPUTS: list[dict] = [
    # code
    {"text": "def hello():\n    print('Hello, world!')", "coarse": "code", "modality": "textual", "code_lang": "python"},
    {"text": "import os\nimport sys\n\ndef main():\n    print(os.getcwd())\n\nif __name__ == '__main__':\n    main()", "coarse": "code", "modality": "textual", "code_lang": "python"},
    {"text": "function greet(name) {\n    return `Hello, ${name}!`;\n}\n\nconsole.log(greet('World'));", "coarse": "code", "modality": "textual", "code_lang": "javascript"},
    {"text": "#include <stdio.h>\n\nint main() {\n    printf(\"hello, world\\n\");\n    return 0;\n}", "coarse": "code", "modality": "textual", "code_lang": "c"},
    {"text": "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}", "coarse": "code", "modality": "textual", "code_lang": "java"},
    {"text": "SELECT id, name, email FROM users WHERE active = 1 ORDER BY name;", "coarse": "code", "modality": "textual", "code_lang": "sql"},
    {"text": "#!/bin/bash\n\nfor f in *.txt; do\n    echo \"Processing $f\"\n    wc -l \"$f\"\ndone", "coarse": "code", "modality": "textual", "code_lang": "bash"},
    {"text": "fn main() {\n    println!(\"Hello, world!\");\n}", "coarse": "code", "modality": "textual", "code_lang": "rust"},
    {"text": "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n    res.json({ message: 'Hello' });\n});\n\napp.listen(3000);", "coarse": "code", "modality": "textual", "code_lang": "javascript"},
    {"text": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"hello world\")\n}", "coarse": "code", "modality": "textual", "code_lang": "go"},
    # markup
    {"text": "<!DOCTYPE html>\n<html>\n<body>\n<h1>Hello</h1>\n</body>\n</html>", "coarse": "markup", "modality": "textual", "code_lang": "html"},
    # text - english
    {"text": "The quick brown fox jumps over the lazy dog. This is a normal English sentence for testing.", "coarse": "text", "modality": "textual", "text_lang": "en"},
    # text - french
    {"text": "Bonjour le monde! Comment allez-vous aujourd'hui? J'espere que tout va bien.", "coarse": "text", "modality": "textual", "text_lang": "fr"},
    # text - hindi
    {"text": "नमस्ते दुनिया! आप कैसे हैं? मुझे उम्मीद है कि सब कुछ ठीक है।", "coarse": "text", "modality": "textual", "text_lang": "hi"},
    # config
    {"text": "{\n  \"name\": \"test\",\n  \"version\": \"1.0.0\",\n  \"description\": \"A test project\"\n}", "coarse": "config", "modality": "textual", "subtype": "json"},
    {"text": "name: test\nversion: 1.0.0\ndependencies:\n  - torch\n  - numpy\n", "coarse": "config", "modality": "textual", "subtype": "yaml"},
    {"text": "DATABASE_URL=postgres://localhost:5432/mydb\nDEBUG=true\nPORT=8080", "coarse": "config", "modality": "textual", "subtype": "env"},
    # error trace
    {"text": "Traceback (most recent call last):\n  File \"test.py\", line 5, in <module>\n    main()\n  File \"test.py\", line 3, in main\n    return 1/0\nZeroDivisionError: division by zero", "coarse": "trace", "modality": "textual"},
    # image (binary header)
    {"text": "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01", "coarse": "image", "modality": "visual"},
    {"text": "GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", "coarse": "image", "modality": "visual"},
    # generic binary
    {"text": "PK\x03\x04\x14\x00\x08\x00\x08\x00\x00\x00!$\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", "coarse": "binary", "modality": "binary"},
]


def _bytes_to_input(text: str, max_bytes: int, device):
    """Convert text to byte-level model input."""
    data = text.encode("utf-8")[:max_bytes]
    ids = torch.zeros(1, len(data), dtype=torch.long, device=device)
    ids[0, :len(data)] = torch.tensor(list(data), dtype=torch.long, device=device)
    mask = torch.ones(1, len(data), dtype=torch.bool, device=device)
    return ids, mask


def evaluate_real_world(model, device, tier="base") -> dict:
    """Evaluate on the 21-item real-world test set."""
    model.eval()
    correct = 0
    total = len(REAL_WORLD_INPUTS)
    details = []
    with torch.no_grad():
        for item in REAL_WORLD_INPUTS:
            text = item["text"]
            ids, mask = _bytes_to_input(text, model.config.max_bytes, device)
            logits = model(ids, mask, tier=tier)
            preds = {}
            for head in ["coarse", "modality", "subtype", "code_lang", "text_lang"]:
                if head in logits:
                    preds[head] = logits[head].argmax(dim=-1).item()
            correct_here = 0
            total_here = 0
            pred_labels = {}
            for head, expected in item.items():
                if head == "text":
                    continue
                if head in preds:
                    predicted = REAL_WORLD_LABEL_MAP[head][preds[head]]
                    pred_labels[head] = predicted
                    if predicted == expected:
                        correct_here += 1
                    total_here += 1
            input_correct = 1 if correct_here == total_here and total_here > 0 else 0
            correct += input_correct
            details.append({
                "text": text[:80],
                "correct": correct_here,
                "total": total_here,
                "predicted": pred_labels,
            })
    return {"accuracy": correct / max(1, total), "correct": correct, "total": total, "details": details}


def run_pipeline(token: str = "", samples_per_lang: int = 30, total_steps: int = 3000, data_path: str = "checkpoints/real_samples.pkl"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Load (or fetch) real code samples
    print("\n========= STEP 1: Load Real Code Samples =========")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                real_samples = data.get("samples", data)
            else:
                real_samples = data
        print(f"Loaded {len(real_samples)} real samples from {data_path}")
    else:
        print("No cached data — fetching from GitHub...")
        from .realdata import build_real_code_dataset as _fetch
        real_samples, stats = _fetch(CODE_LANG_LABELS, samples_per_lang, token)
        os.makedirs("checkpoints", exist_ok=True)
        with open(data_path, "wb") as f:
            pickle.dump(real_samples, f)
        print(f"Fetched and saved {len(real_samples)} samples")

    total_real = len(real_samples)

    # Step 2: Evaluate current best.pt
    print("\n========= STEP 2: Evaluate Current best.pt =========")
    model = PicoType(PicoTypeConfig(max_bytes=2048)).to(device)
    ckpt_path = "checkpoints/best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
        print(f"Loaded {ckpt_path}")
        import time as _time
        t0 = _time.time()
        before = evaluate_real_world(model, device)
        elapsed = _time.time() - t0
        print(f"Before fine-tuning: {before['accuracy']*100:.1f}% ({before['correct']}/{before['total']}) in {elapsed:.1f}s")
    else:
        print("No best.pt found, starting from scratch")
        before = {"accuracy": 0.0, "correct": 0, "total": len(REAL_WORLD_INPUTS)}

    # Step 3: Fine-tune with real data
    print(f"\n========= STEP 3: Fine-tune {total_steps} steps =========")
    cfg = TrainConfig(
        lr=2e-4,
        total_steps=total_steps,
        batch_size=16,
        warmup_steps=100,
        train_size=max(10000, total_real * 3),
        eval_size=500,
        log_every=50,
        output_dir="checkpoints",
        train_tiers=("base",),
        tier="base",
        resume_from=ckpt_path if os.path.exists(ckpt_path) else "",
    )
    cfg.device = device

    model.train()
    criterion = MultiTaskLoss(cfg.head_weights)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "linears" not in n], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if "linears" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)

    gen = SyntheticGenerator(seed=cfg.seed)
    eval_gen = SyntheticGenerator(seed=cfg.seed + 1)
    eval_ds = SyntheticDataset(eval_gen, cfg.eval_size)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=cfg.batch_size, collate_fn=collate_fn, num_workers=0,
    )

    step = 0
    best_loss = float("inf")
    print(f"Training {total_steps} steps with {total_real} real samples per epoch...")
    while step < total_steps:
        # Synthetic batch
        synth_batch = [gen() for _ in range(cfg.batch_size)]
        collated = collate_fn(synth_batch)
        ids = collated["input_ids"].to(device)
        mask = collated["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in collated["labels"].items()}

        # Real code batch
        real_batch_size = max(1, int(cfg.batch_size * 0.3))
        real_idx = (step * real_batch_size) % max(1, len(real_samples))
        real_batch = real_samples[real_idx:real_idx + real_batch_size]
        if len(real_batch) < real_batch_size:
            real_batch = real_samples[:real_batch_size]

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss, comps = multi_tier_loss(model, ids, mask, labels, criterion, cfg.train_tiers)

        if real_batch:
            real_collated = collate_fn(real_batch)
            real_ids = real_collated["input_ids"].to(device)
            real_mask = real_collated["attention_mask"].to(device)
            real_labels = {k: v.to(device) for k, v in real_collated["labels"].items()}
            real_loss, real_comps = multi_tier_loss(model, real_ids, real_mask, real_labels, criterion, cfg.train_tiers)
            loss = loss + real_loss
            for k, v in real_comps.items():
                comps[f"real_{k}"] = v

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0:
            parts = [f"step={step:5d} lr={lr:.6f} loss={loss.item():.4f}"]
            for h in ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk"):
                if h in comps:
                    parts.append(f"{h}={comps[h]:.4f}")
            print("  ".join(parts))

        step += 1

        # Eval every 500 steps
        if step % 500 == 0 or step == total_steps:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            with torch.no_grad():
                for eb in eval_loader:
                    eids = eb["input_ids"].to(device)
                    emask = eb["attention_mask"].to(device)
                    elbls = {k: v.to(device) for k, v in eb["labels"].items()}
                    _, ecomps = multi_tier_loss(model, eids, emask, elbls, criterion, cfg.train_tiers)
                    eval_loss += ecomps.get("total", 0)
                    eval_batches += 1
            avg_eval = eval_loss / max(1, eval_batches)
            rw = evaluate_real_world(model, device)
            print(f"  EVAL step={step}  synth_loss={avg_eval:.4f}  real_world={rw['accuracy']*100:.1f}%")
            if avg_eval < best_loss:
                best_loss = avg_eval
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": avg_eval,
                    "real_world_accuracy": rw["accuracy"],
                }, "checkpoints/pipeline_best.pt")
                print(f"  New best saved! (eval_loss={avg_eval:.4f})")
            model.train()

    # Step 4: Final evaluation
    print("\n========= STEP 4: Final Evaluation =========")
    final_path = "checkpoints/pipeline_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, final_path)
    print(f"Saved final model to {final_path}")

    model.eval()
    before_acc = before["accuracy"]
    rw = evaluate_real_world(model, device)
    print(f"\nBefore: {before_acc*100:.1f}%")
    print(f"After:  {rw['accuracy']*100:.1f}%")
    print(f"Delta:  {(rw['accuracy'] - before_acc)*100:+.1f}%")
    print("\nPer-input details:")
    for d in rw["details"]:
        status = "✓" if d["correct"] == d["total"] else "✗"
        print(f"  {status} [{d['correct']}/{d['total']}] {d['text'][:60]}")

    # If better than current best.pt, replace it
    old_best = before_acc
    if rw["accuracy"] > old_best:
        print(f"\n!!! New model beats best.pt ({rw['accuracy']*100:.1f}% vs {old_best*100:.1f}%) !!!")
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "real_world_accuracy": rw["accuracy"],
        }, "checkpoints/best.pt")
        print("Replaced checkpoints/best.pt!")
    else:
        print(f"\nNew model did not beat best.pt ({rw['accuracy']*100:.1f}% vs {old_best*100:.1f}%)")

    return rw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="", help="GitHub token")
    parser.add_argument("--samples-per-lang", type=int, default=30)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--data", default="checkpoints/real_samples.pkl")
    args = parser.parse_args()
    run_pipeline(token=args.token, samples_per_lang=args.samples_per_lang, total_steps=args.total_steps, data_path=args.data)
