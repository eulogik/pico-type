"""Generate config.json (Hugging Face download-tracking query file) from canonical labels + metadata."""

import json

from model.pico_type.arch import PicoTypeConfig, TIERS
from model.pico_type.labels import (
    COARSE_LABELS,
    MODALITY_LABELS,
    SUBTYPE_LABELS,
    CODE_LANG_LABELS,
    TEXT_LANG_LABELS,
    FILE_MIME_LABELS,
    RISK_LABELS,
)

LABEL_TABLES = {
    "coarse": COARSE_LABELS,
    "modality": MODALITY_LABELS,
    "subtype": SUBTYPE_LABELS,
    "code_lang": CODE_LANG_LABELS,
    "text_lang": TEXT_LANG_LABELS,
    "file_mime": FILE_MIME_LABELS,
    "risk": RISK_LABELS,
}

LABEL_MAP = {head: {label: i for i, label in enumerate(labels)} for head, labels in LABEL_TABLES.items()}

cfg = PicoTypeConfig()

config = {
    "model_type": "pico_type",
    "architectures": ["PicoType"],
    "model_name": "pico-type",
    "model_id": "eulogik/pico-type",
    "vocab_size": 256,
    "embed_dim": cfg.embed_dim,
    "trunk_dim": cfg.trunk_dim,
    "max_bytes": cfg.max_bytes,
    "tiers": TIERS,
    "tier_dims": TIERS,
    "heads": {head: len(labels) for head, labels in LABEL_TABLES.items()},
    "opset": 18,
    "label_maps": LABEL_MAP,
    "benchmarks": {
        "code_lang": 60.08,
        "text_lang": 98.33,
    },
    "paper": "https://arxiv.org/abs/2608.14658",
    "license": "apache-2.0",
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

print(f"wrote config.json: {len(json.dumps(config))} bytes")