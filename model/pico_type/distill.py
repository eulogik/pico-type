"""pico-type: knowledge distillation pipeline (optional, teacher-dependent)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .arch import PicoType
from .data import (
    IGNORE_INDEX,
    MixedDataset,
    RealDataset,
    SyntheticDataset,
    SyntheticGenerator,
)
from .labels import (
    HEAD_NUM_CLASSES,
)
from .train import (
    TrainConfig,
    collate_fn,
    get_lr,
)


@dataclass
class DistillConfig:
    teacher_model: str = ""
    teacher_cache_dir: str = ""
    alpha: float = 0.7
    temperature: float = 2.0
    freeze_teachers: bool = True
    backbone_only: bool = False
    real_code_path: str = ""
    real_text_path: str = ""
    real_ratio: float = 0.5
    train_config: TrainConfig = field(default_factory=TrainConfig)


class DistillDataset(SyntheticDataset):
    """Dataset that pairs synthetic samples with teacher soft labels."""

    def __init__(
        self,
        generator: SyntheticGenerator,
        size: int,
        teachers: dict[str, nn.Module] | None = None,
        temperature: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(generator, size)
        self.teachers = teachers
        self.temperature = temperature
        self.device = device
        self._soft_labels: dict[str, torch.Tensor] | None = None

    def _ensure_soft(self) -> dict[str, torch.Tensor]:
        if self._soft_labels is not None:
            return self._soft_labels
        if not self.teachers:
            return {}
        samples = self._ensure()
        soft: dict[str, list[torch.Tensor]] = {}
        for sample in samples:
            text = sample.data.decode("utf-8", errors="replace")
            for head, teacher in self.teachers.items():
                if head not in soft:
                    soft[head] = []
                try:
                    logits = self._run_teacher(teacher, text, head)
                except Exception:
                    logits = torch.full((HEAD_NUM_CLASSES[head],), float("-inf"))
                soft[head].append(logits)
        self._soft_labels = {h: torch.stack(v) for h, v in soft.items()}
        return self._soft_labels

    def _run_teacher(self, teacher: nn.Module, text: str, head: str) -> torch.Tensor:
        teacher.eval()
        with torch.no_grad():
            inputs = self._tokenize(text, teacher, head)
            outputs = teacher(**inputs)
            logits = outputs.logits[0]
        return logits.detach().cpu()

    @staticmethod
    def _tokenize(text: str, teacher: nn.Module, head: str) -> dict:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("install transformers: pip install transformers")
        name = teacher.config._name_or_path
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or "[PAD]"
        return tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)


class DistillLoss(nn.Module):
    """Combines hard-label CE with teacher KL divergence."""

    def __init__(
        self,
        weights: dict[str, float],
        alpha: float = 0.7,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.weights = weights
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=IGNORE_INDEX)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        teacher_logits: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros(1, device=next(iter(student_logits.values())).device)
        individual: dict[str, float] = {}
        for head in student_logits:  # noqa: PLC0206
            w = self.weights.get(head, 1.0)
            if head == "risk":
                hard_loss = self.bce(student_logits[head], labels[head])
            else:
                hard_loss = self.ce(student_logits[head], labels[head])
            if teacher_logits and head in teacher_logits and teacher_logits[head] is not None:
                s_log = F.log_softmax(student_logits[head] / self.temperature, dim=-1)
                t_soft = F.softmax(teacher_logits[head] / self.temperature, dim=-1)
                kd = self.kl(s_log, t_soft) * self.temperature ** 2
                loss = self.alpha * hard_loss + (1 - self.alpha) * kd
            else:
                loss = hard_loss
            total = total + w * loss
            individual[head] = loss.detach().item()
        individual["total"] = total.detach().item()
        return total, individual


def build_teachers(
    device: torch.device,
    cache_dir: str = "",
) -> dict[str, nn.Module]:
    """Download and load per-head teacher models from HuggingFace."""
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        raise ImportError("install transformers: pip install transformers")

    kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    teachers: dict[str, nn.Module] = {}

    try:
        teachers["coarse"] = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-small", num_labels=HEAD_NUM_CLASSES["coarse"], **kwargs
        ).to(device).eval()
    except Exception:
        pass

    try:
        teachers["modality"] = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-small", num_labels=HEAD_NUM_CLASSES["modality"], **kwargs
        ).to(device).eval()
    except Exception:
        pass

    try:
        teachers["code_lang"] = AutoModelForSequenceClassification.from_pretrained(
            "huggingface/CodeBERTa-language-id", **kwargs
        ).to(device).eval()
    except Exception:
        pass

    try:
        teachers["text_lang"] = AutoModelForSequenceClassification.from_pretrained(
            "papluca/xlm-roberta-base-language-detection", **kwargs
        ).to(device).eval()
    except Exception:
        pass

    return teachers


def distill_train(config: DistillConfig) -> DistillConfig:
    cfg = config.train_config

    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(cfg.device)

    teachers = build_teachers(device, config.teacher_cache_dir) if config.teacher_model else {}
    print(f"Teachers loaded: {list(teachers)}")

    gen = SyntheticGenerator(seed=cfg.seed)
    synt_ds = SyntheticDataset(gen, cfg.train_size)
    train_ds = synt_ds
    if config.real_code_path or config.real_text_path:
        real_ds = RealDataset(config.real_code_path, config.real_text_path, cfg.seed)
        train_ds = MixedDataset(synt_ds, real_ds, config.real_ratio, cfg.seed)
        print(f"Mixed dataset: {len(train_ds)} steps (real_ratio={config.real_ratio})")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = PicoType(cfg.model_config).to(device)
    criterion = DistillLoss(cfg.head_weights, config.alpha, config.temperature)

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "linears" not in n], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if "linears" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)

    os.makedirs(cfg.output_dir, exist_ok=True)
    step = 0
    train_ds._ensure_soft()

    while step < cfg.total_steps:
        model.train()
        for batch in train_loader:
            if step >= cfg.total_steps:
                break

            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            optimizer.zero_grad()
            logits = model(ids, mask, tier=cfg.tier)
            teacher_logits = _batch_teacher_logits(train_ds, step, cfg.batch_size, device, teachers)
            loss, comps = criterion(logits, labels, teacher_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if step % cfg.log_every == 0:
                parts = [f"step={step:5d} lr={lr:.6f}"]
                for h, v in comps.items():
                    parts.append(f"{h}={v:.4f}")
                print("  ".join(parts))

            step += 1

    torch.save({"step": step, "model_state_dict": model.state_dict()}, os.path.join(cfg.output_dir, "distill_final.pt"))
    return config


def _batch_teacher_logits(
    ds: DistillDataset, step: int, batch_size: int, device: torch.device, teachers: dict[str, nn.Module]
) -> dict[str, torch.Tensor] | None:
    if not teachers:
        return None
    soft = ds._ensure_soft()
    if not soft:
        return None
    start = (step * batch_size) % len(ds)
    end = min(start + batch_size, len(ds))
    return {head: t[start:end].to(device) for head, t in soft.items()}
