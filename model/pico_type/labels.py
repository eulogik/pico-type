"""pico-type: label vocabularies and output decoder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


COARSE_LABELS = [
    "text", "code", "link", "image", "file", "config",
    "markup", "data", "error", "secret", "archive", "binary",
]

MODALITY_LABELS = [
    "textual", "binary_image", "binary_archive", "binary_executable",
    "binary_document", "binary_audio", "binary_video", "binary_other",
]

SUBTYPE_LABELS = [
    "json", "yaml", "toml", "ini", "csv", "tsv", "xml",
    "html", "markdown", "rst", "asciidoc", "tex",
    "sql", "graphql", "protobuf", "msgpack",
    "log", "diff", "patch", "env",
    "shell", "makefile", "dockerfile", "gitignore",
]

CODE_LANG_LABELS = [
    "python", "javascript", "typescript", "jsx", "tsx",
    "java", "kotlin", "scala", "groovy", "clojure",
    "c", "cpp", "csharp", "fsharp", "objectivec",
    "go", "rust", "zig",
    "ruby", "php", "perl", "lua", "tcl",
    "swift", "dart", "julia", "nim", "crystal",
    "haskell", "ocaml", "elm", "erlang", "elixir",
    "lisp", "scheme", "racket",
    "r", "matlab", "octave", "sas", "stata",
    "sql", "plsql", "tsql",
    "html", "css", "scss", "sass", "less",
    "bash", "zsh", "fish", "powershell",
    "vim", "fortran", "cobol", "ada", "pascal",
    "delphi", "vb", "prolog", "vhdl",
]

assert len(CODE_LANG_LABELS) == 62, f"expected 62 code langs, got {len(CODE_LANG_LABELS)}"

TEXT_LANG_LABELS = [
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da",
    "fi", "pl", "cs", "sk", "hu", "ro", "el", "tr",
    "ru", "uk", "bg", "sr", "hr",
    "zh", "ja", "ko", "vi", "th", "id", "ms",
]

assert len(TEXT_LANG_LABELS) == 30, f"expected 30 text langs, got {len(TEXT_LANG_LABELS)}"

FILE_MIME_LABELS = [
    "application/pdf", "application/zip", "application/gzip", "application/x-tar",
    "application/x-7z-compressed", "application/x-rar-compressed", "application/x-bzip2",
    "application/x-xz", "application/json", "application/xml", "application/yaml",
    "application/octet-stream", "application/x-executable", "application/x-mach-binary",
    "application/x-elf", "application/x-deb", "application/x-rpm",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
    "application/msword", "application/rtf", "application/epub+zip",
    "application/x-ndjson",
    "text/plain", "text/csv", "text/html", "text/xml", "text/markdown",
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml",
    "image/bmp", "image/tiff", "image/heic", "image/heif", "image/avif",
    "image/x-icon", "image/vnd.adobe.photoshop",
    "video/mp4", "video/webm", "video/x-matroska", "video/quicktime",
    "video/x-msvideo", "video/x-flv", "video/x-mpeg",
    "audio/mpeg", "audio/ogg", "audio/wav", "audio/flac", "audio/aac",
    "audio/x-m4a", "audio/webm", "audio/midi",
    "font/ttf", "font/otf", "font/woff", "font/woff2",
    "application/x-sqlite3", "application/x-parquet",
    "application/x-protobuf", "application/x-flatbuffers",
    "application/x-cpio", "application/x-iso9660-image",
    "application/vnd.android.package-archive", "application/x-jar",
    "application/x-python-bytecode", "application/x-archive",
    "application/pgp-encrypted", "application/pgp-signature",
    "application/x-x509-ca-cert", "application/x-pem-file",
    "application/vnd.tcpdump.pcap",
    "application/java-vm",
    "application/x-matlab-data",
    "application/x-shockwave-flash",
    "application/x-font-ttf", "application/x-font-otf",
    "application/wasm", "application/x-ruby",
    "application/javascript", "application/ecmascript",
    "application/x-bittorrent", "application/x-dvi",
    "chemical/x-mdl-sdfile",
    "application/x-lzma",
]

assert len(FILE_MIME_LABELS) == 90, f"expected 90 file mimes, got {len(FILE_MIME_LABELS)}"

RISK_LABELS = ["api_key", "jwt", "ssh_key", "password", "email", "phone"]

assert len(RISK_LABELS) == 6, f"expected 6 risk labels, got {len(RISK_LABELS)}"

UNDETECTED = "undetected"

ALL_HEADS = ("coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk")

HEAD_NUM_CLASSES: Dict[str, int] = {
    "coarse": len(COARSE_LABELS),
    "modality": len(MODALITY_LABELS),
    "subtype": len(SUBTYPE_LABELS),
    "code_lang": len(CODE_LANG_LABELS),
    "text_lang": len(TEXT_LANG_LABELS),
    "file_mime": len(FILE_MIME_LABELS),
    "risk": len(RISK_LABELS),
}

SUBTYPE_GATED_BY = {"config", "markup", "data"}
CODE_LANG_GATED_BY = {"code"}
TEXT_LANG_GATED_BY = {"text"}
FILE_MIME_GATED_BY = {
    "image", "file", "archive", "binary",
    "binary_image", "binary_archive", "binary_executable",
    "binary_document", "binary_audio", "binary_video", "binary_other",
}


def _pick_with_threshold(
    logits: torch.Tensor,
    labels: list[str],
    threshold: float,
) -> tuple[Optional[str], float]:
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    if conf.item() < threshold:
        return UNDETECTED, conf.item()
    return labels[idx.item()], conf.item()


def decode_output(
    logits: Dict[str, torch.Tensor],
    tier: str = "base",
    undetected_threshold: float = 0.4,
    risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    coarse_idx = int(logits["coarse"].argmax(dim=-1).item())
    coarse = COARSE_LABELS[coarse_idx]
    coarse_conf = float(torch.softmax(logits["coarse"], dim=-1).max().item())

    modality_idx = int(logits["modality"].argmax(dim=-1).item())
    modality = MODALITY_LABELS[modality_idx]
    modality_conf = float(torch.softmax(logits["modality"], dim=-1).max().item())

    subtype: Optional[str] = None
    if coarse in SUBTYPE_GATED_BY:
        subtype, _ = _pick_with_threshold(
            logits["subtype"], SUBTYPE_LABELS, undetected_threshold,
        )

    code_lang: Optional[str] = None
    if coarse in CODE_LANG_GATED_BY:
        code_lang, _ = _pick_with_threshold(
            logits["code_lang"], CODE_LANG_LABELS, undetected_threshold,
        )

    text_lang: Optional[str] = None
    if coarse in TEXT_LANG_GATED_BY:
        text_lang, _ = _pick_with_threshold(
            logits["text_lang"], TEXT_LANG_LABELS, undetected_threshold,
        )

    file_mime: Optional[str] = None
    if coarse in FILE_MIME_GATED_BY or modality in FILE_MIME_GATED_BY:
        file_mime, _ = _pick_with_threshold(
            logits["file_mime"], FILE_MIME_LABELS, undetected_threshold,
        )

    risk_logits = logits["risk"]
    risk_probs = torch.sigmoid(risk_logits).squeeze(0)
    risk_flags = [
        RISK_LABELS[i] for i in range(len(RISK_LABELS))
        if float(risk_probs[i].item()) >= risk_threshold
    ]

    return {
        "coarse": coarse,
        "modality": modality,
        "subtype": subtype,
        "code_language": code_lang,
        "text_language": text_lang,
        "file_mime": file_mime,
        "risk_flags": risk_flags,
        "confidence": coarse_conf,
        "modality_confidence": modality_conf,
        "model_tier": tier,
    }


def label_for(head: str, idx: int) -> str:
    table = {
        "coarse": COARSE_LABELS,
        "modality": MODALITY_LABELS,
        "subtype": SUBTYPE_LABELS,
        "code_lang": CODE_LANG_LABELS,
        "text_lang": TEXT_LANG_LABELS,
        "file_mime": FILE_MIME_LABELS,
        "risk": RISK_LABELS,
    }
    return table[head][idx]
