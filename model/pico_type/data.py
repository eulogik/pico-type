"""pico-type: synthetic data generator for multi-head training."""

from __future__ import annotations

import base64
import hashlib
import random
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .labels import (
    COARSE_LABELS, MODALITY_LABELS, SUBTYPE_LABELS,
    CODE_LANG_LABELS, TEXT_LANG_LABELS, FILE_MIME_LABELS, RISK_LABELS,
    HEAD_NUM_CLASSES,
)

IGNORE_INDEX = -100
MIN_BYTES = 32
MAX_BYTES = 1023

_COARSE = {name: i for i, name in enumerate(COARSE_LABELS)}
_MODALITY = {name: i for i, name in enumerate(MODALITY_LABELS)}
_SUBTYPE = {name: i for i, name in enumerate(SUBTYPE_LABELS)}
_CODE = {name: i for i, name in enumerate(CODE_LANG_LABELS)}
_TEXT = {name: i for i, name in enumerate(TEXT_LANG_LABELS)}
_MIME = {name: i for i, name in enumerate(FILE_MIME_LABELS)}
_RISK = {name: i for i, name in enumerate(RISK_LABELS)}

_LANG_GROUPS: Dict[str, str] = {
    # python-like (indent-based)
    "python": "python", "ruby": "python", "nim": "python", "julia": "python", "crystal": "python",
    "elixir": "python", "cobol": "python",
    # C-like (braces, semicolons)
    "c": "c", "cpp": "c", "csharp": "c", "java": "c", "kotlin": "c",
    "groovy": "c", "objectivec": "c", "scala": "c", "dart": "c",
    "verilog": "c", "vhdl": "c",
    # C++-derived
    "zig": "c", "carbon": "c",
    # Go-like
    "go": "go",
    # Rust-like
    "rust": "rust",
    # Lisp-like
    "lisp": "lisp", "scheme": "lisp", "racket": "lisp", "clojure": "lisp",
    "commonlisp": "lisp", "emacs": "lisp",
    # Haskell-like (functional)
    "haskell": "haskell", "ocaml": "haskell", "elm": "haskell",
    "erlang": "erlang",
    # JS-like
    "javascript": "js", "typescript": "js", "jsx": "js", "tsx": "js",
    "action": "js",
    # PHP/Perl (sigil-based)
    "php": "php", "perl": "php",
    # Lua
    "lua": "lua",
    # Swift
    "swift": "swift",
    # TCL
    "tcl": "tcl",
    # R-like
    "r": "r", "matlab": "r", "octave": "r", "sas": "r", "stata": "r",
    # Racket goes to lisp
    # Fortran/Ada/Pascal (verbose, begin-end)
    "fortran": "fortran", "ada": "fortran", "pascal": "fortran",
    "delphi": "fortran",
    # ABAP-like
    "abap": "abap",
    # Prolog
    "prolog": "prolog",
    # VB
    "vb": "vb",
    # markup-based
    "html": "markup", "css": "markup", "scss": "markup", "sass": "markup",
    "less": "markup",
    # shell
    "bash": "sh", "zsh": "sh", "fish": "sh", "powershell": "sh",
    "shell": "sh",
    # vim
    "vim": "vim",
    # SQL
    "sql": "sql", "plsql": "sql", "tsql": "sql",
    "fsharp": "c",
}

_CODE_TEMPLATES: Dict[str, List[str]] = {
    "python": [
        "def ${func}(${args}):\n    \"\"\"docstring\"\"\"\n    ${val}\n    return ${val}\n",
        "for ${loopvar} in ${ident}:\n    ${ident}(${args})\n    break\n",
        "import ${mod}\nfrom ${mod} import ${name}\n\nclass ${cls}(object):\n    def __init__(self, ${args}):\n        self.${attr} = ${val}\n",
    ],
    "c": [
        "#include <stdio.h>\n\nint ${func}(int ${var}) {\n    return ${val};\n}\n",
        "typedef struct { int ${var}; char ${attr}; } ${cls};\n\nint main() {\n    ${cls} ${ident} = {0};\n    return 0;\n}\n",
        "// ${msg}\n#ifndef ${mod}\n#define ${mod}\nvoid ${func}(${args}) {}\n#endif\n",
    ],
    "go": [
        "package main\n\nfunc ${func}(${args}) ${ident} {\n    return ${val}\n}\n",
        "type ${cls} struct {\n    ${var} int\n}\n\nfunc (t *${cls}) ${func}() ${ident} {\n    return t.${var}\n}\n",
    ],
    "rust": [
        "fn ${func}(${args}) -> ${ident} {\n    ${val}\n}\n\nfn main() {\n    let ${var} = ${func}(${args});\n}\n",
        "#[derive(Debug)]\nstruct ${cls} {\n    ${var}: ${ident},\n}\n\nimpl ${cls} {\n    fn ${func}() -> Self { Self { ${var}: ${val} } }\n}\n",
    ],
    "lisp": [
        "(defun ${func} (${args})\n  \"${msg}\"\n  ${val})\n",
        "(defparameter *${var}* ${val})\n\n(defun ${ident} ()\n  (format t \"~a\" *${var}*))\n",
    ],
    "haskell": [
        "${func} :: ${ident} -> ${ident}\n${func} ${var} = ${val}\n\nmain :: IO ()\nmain = print $ ${func} ${val}\n",
        "data ${cls} = ${cls} { ${var} :: ${ident} } deriving (Show)\n\n${func} :: ${cls} -> ${ident}\n${func} (${cls} ${var}) = ${var}\n",
    ],
    "erlang": [
        "-module(${mod}).\n-export([${func}/1]).\n\n${func}(${args}) -> ${val}.\n",
    ],
    "js": [
        "function ${func}(${args}) {\n    return ${val};\n}\n\n${func}(${val});\n",
        "const ${var} = (${args}) => {\n    console.log(${var});\n    return ${val};\n};\n",
        "class ${cls} {\n    constructor(${args}) {\n        this.${attr} = ${val};\n    }\n}\n\nexport default ${cls};\n",
    ],
    "php": [
        "<?php\nfunction ${func}(${args}) {\n    $${var} = ${val};\n    return $${var};\n}\n?>\n",
    ],
    "lua": [
        "function ${func}(${args})\n    local ${var} = ${val}\n    return ${var}\nend\n\n${func}(${val})\n",
    ],
    "swift": [
        "func ${func}(${args}) -> ${ident} {\n    return ${val}\n}\n\nlet ${var} = ${func}(${val})\n",
    ],
    "tcl": [
        "proc ${func} {${args}} {\n    set ${var} ${val}\n    return $${var}\n}\n\n${func} ${val}\n",
    ],
    "r": [
        "${func} <- function(${args}) {\n    ${var} <- ${val}\n    return(${var})\n}\n\n${func}(${val})\n",
    ],
    "fortran": [
        "program ${mod}\n  implicit none\n  integer :: ${var}\n  ${var} = ${val}\n  print *, ${var}\nend program ${mod}\n",
    ],
    "abap": [
        "REPORT ${mod}.\nFORM ${func} USING ${args}.\n  DATA: ${var} TYPE i.\n  ${var} = ${val}.\nENDFORM.\n",
    ],
    "prolog": [
        "${func}(${args}) :- ${val}.\n\n:- ${func}(${val}).\n",
    ],
    "vb": [
        "Module ${mod}\n    Sub ${func}(${args})\n        Dim ${var} As Integer = ${val}\n    End Sub\nEnd Module\n",
    ],
    "markup": [
        "<!DOCTYPE html>\n<html>\n<head><title>${msg}</title></head>\n<body>\n<div class=\"${ident}\">${msg}</div>\n</body>\n</html>\n",
        ".${cls} { background-color: ${var}; }\n\n#${ident} { margin: ${val}px; }\n",
    ],
    "sh": [
        "#!/bin/bash\n\n${func}() {\n    local ${var}=${val}\n    echo $${var}\n}\n\n${func}\n",
    ],
    "vim": [
        "\" ${msg}\nfunction! ${func}(${args}) abort\n    return ${val}\nendfunction\n",
    ],
    "sql": [
        "CREATE TABLE ${cls} ( ${var} INT, ${attr} VARCHAR(255) );\n\nSELECT * FROM ${cls} WHERE ${var} = ${val};\n",
    ],
}


@dataclass
class Sample:
    data: bytes
    coarse: int = IGNORE_INDEX
    modality: int = IGNORE_INDEX
    subtype: int = IGNORE_INDEX
    code_lang: int = IGNORE_INDEX
    text_lang: int = IGNORE_INDEX
    file_mime: int = IGNORE_INDEX
    risk: List[int] = field(default_factory=list)

    def label_dict(self) -> Dict[str, int]:
        result = {}
        for head in (
            "coarse", "modality", "subtype",
            "code_lang", "text_lang", "file_mime",
        ):
            v = getattr(self, head)
            result[head] = v
        result["risk"] = self.risk
        return result

    def check(self) -> None:
        assert MIN_BYTES <= len(self.data) <= MAX_BYTES, f"bad size {len(self.data)}"
        for head in ("coarse", "modality"):
            v: int = getattr(self, head)
            assert 0 <= v < HEAD_NUM_CLASSES[head], f"bad {head}={v}"
        for head in ("subtype", "code_lang", "text_lang", "file_mime"):
            v = getattr(self, head)
            if v != IGNORE_INDEX:
                assert 0 <= v < HEAD_NUM_CLASSES[head], f"bad {head}={v}"
        for r in self.risk:
            assert 0 <= r < HEAD_NUM_CLASSES["risk"], f"bad risk={r}"


class SyntheticGenerator:
    """Generates balanced synthetic samples for all 7 heads."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def __call__(self) -> Sample:
        bucket = self.rng.choice([
            "code", "text", "config", "markup", "data",
            "link", "error", "image", "file", "secret", "archive", "binary",
        ])
        fn = getattr(self, f"_gen_{bucket}")
        sample = fn()
        if len(sample.data) > MAX_BYTES:
            sample = Sample(data=sample.data[:MAX_BYTES], **sample.label_dict())
        if len(sample.data) < MIN_BYTES:
            return self()
        sample.check()
        return sample

    def _rng_ident(self, max_len: int = 12) -> str:
        return self.rng.choice(string.ascii_lowercase) + "".join(
            self.rng.choices(string.ascii_lowercase + string.digits, k=self.rng.randint(1, max_len - 1))
        )

    def _rng_val(self) -> str:
        return self.rng.choice([
            str(self.rng.randint(0, 100)),
            f'"{self._rng_ident(8)}"',
            str(self.rng.random()),
            "True", "False", "None",
        ])

    def _rng_pad(self, target: int) -> bytes:
        data = self._rng_ident(20).encode()
        while len(data) < target:
            data += b" " + self._rng_ident(20).encode()
        return data[:target]

    # ---- code ----
    def _gen_code(self) -> Sample:
        lang = self.rng.choice(CODE_LANG_LABELS)
        lang_idx = _CODE[lang]

        group = _LANG_GROUPS.get(lang, "python")
        templates = _CODE_TEMPLATES[group]
        tmpl = self.rng.choice(templates)

        defs_seen: set = set()

        def _fill(t: str) -> str:
            def _repl(m: re.Match) -> str:
                kind = m.group(1)
                key = self._rng_ident(10)
                while key in defs_seen:
                    key = self._rng_ident(10)
                defs_seen.add(key)
                return {
                    "ident": key,
                    "val": self._rng_val(),
                    "cls": key.title() + self._rng_ident(5),
                    "mod": key,
                    "name": key,
                    "var": key,
                    "args": ", ".join(self._rng_ident(6) for _ in range(self.rng.randint(0, 3))),
                    "func": self._rng_ident(8),
                    "attr": key,
                    "msg": self._rng_ident(15),
                    "loopvar": self._rng_ident(5),
                }.get(kind, key)
            return re.sub(r"\$\{(\w+)\}", _repl, t)

        body = _fill(tmpl)
        data = body.encode("utf-8")

        return Sample(
            data=data,
            coarse=_COARSE["code"],
            modality=_MODALITY["textual"],
            code_lang=lang_idx,
        )

    # ---- text ----
    def _gen_text(self) -> Sample:
        lang = self.rng.choice(TEXT_LANG_LABELS)
        lang_idx = _TEXT[lang]
        words = _TEXT_WORDS.get(lang, _TEXT_WORDS["en"])
        sentence_count = self.rng.randint(1, 5)
        sentences = []
        for _ in range(sentence_count):
            wc = self.rng.randint(5, 15)
            sent = " ".join(self.rng.choice(words) for _ in range(wc))
            sent = sent[0].upper() + sent[1:] + "."
            sentences.append(sent)
        data = " ".join(sentences).encode("utf-8")

        # risk detection on text
        risk_indices = self._detect_risk(data)
        return Sample(
            data=data,
            coarse=_COARSE["text"],
            modality=_MODALITY["textual"],
            text_lang=lang_idx,
            risk=risk_indices,
        )

    # ---- config ----
    def _gen_config(self) -> Sample:
        subtype = self.rng.choice(["json", "yaml", "toml", "ini", "env"])
        entries = self.rng.randint(3, 10)
        if subtype == "json":
            parts = ["{"]
            for _ in range(entries):
                k = self._rng_ident(8)
                v = self._rng_val()
                parts.append(f'  "{k}": {v},')
            parts.append("}")
            data = "\n".join(parts).encode()
        elif subtype == "yaml":
            parts = []
            for _ in range(entries):
                parts.append(f"{self._rng_ident(8)}: {self._rng_val()}")
            data = "\n".join(parts).encode()
        elif subtype == "toml":
            parts = [f"[{self._rng_ident(8)}]"]
            for _ in range(entries):
                parts.append(f"{self._rng_ident(8)} = {self._rng_val()}")
            data = "\n".join(parts).encode()
        elif subtype == "ini":
            parts = [f"[{self._rng_ident(8)}]"]
            for _ in range(entries):
                parts.append(f"{self._rng_ident(8)}={self._rng_val()}")
            data = "\n".join(parts).encode()
        else:  # env
            parts = []
            for _ in range(entries):
                parts.append(f"{self._rng_ident(8).upper()}={self._rng_val()}")
            data = "\n".join(parts).encode()
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        return Sample(
            data=data,
            coarse=_COARSE["config"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE.get(subtype),
        )

    # ---- markup ----
    def _gen_markup(self) -> Sample:
        fmt = self.rng.choice(["xml", "html", "markdown", "rst", "asciidoc", "tex"])
        if fmt == "html":
            lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head><title>Page</title></head>",
                "<body>",
            ]
            for _ in range(self.rng.randint(2, 6)):
                tag = self.rng.choice(["p", "div", "span", "h2", "h3", "li", "a"])
                if tag == "a":
                    lines.append(f'<a href="https://{self._rng_ident(8)}.com">{self._rng_ident(10)}</a>')
                else:
                    lines.append(f"<{tag}>{self._rng_ident(20)}</{tag}>")
            lines.extend(["</body>", "</html>"])
            data = "\n".join(lines).encode()
        elif fmt == "xml":
            lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<root>"]
            for _ in range(self.rng.randint(3, 8)):
                tag = self._rng_ident(8)
                lines.append(f"  <{tag}>{self._rng_ident(20)}</{tag}>")
            lines.append("</root>")
            data = "\n".join(lines).encode()
        elif fmt == "markdown":
            lines = [f"# {self._rng_ident(20).title()}"]
            for _ in range(self.rng.randint(3, 8)):
                item = f"- **{self._rng_ident(8)}**: {self._rng_ident(20)}."
                lines.append(item)
            data = "\n".join(lines).encode()
        elif fmt == "rst":
            title = self._rng_ident(20).title()
            lines = [title, "=" * len(title), ""]
            for _ in range(self.rng.randint(3, 8)):
                lines.append(f"* {self._rng_ident(8)} -- {self._rng_ident(20)}")
            data = "\n".join(lines).encode()
        elif fmt == "asciidoc":
            lines = [f"= {self._rng_ident(20).title()}", ""]
            for _ in range(self.rng.randint(3, 8)):
                lines.append(f"*{self._rng_ident(8)}*:: {self._rng_ident(20)}")
            data = "\n".join(lines).encode()
        else:  # tex
            lines = [
                "\\documentclass{article}",
                "\\begin{document}",
                f"\\section{{{self._rng_ident(12).title()}}}",
            ]
            for _ in range(self.rng.randint(3, 8)):
                lines.append(f"\\textbf{{{self._rng_ident(8)}}} {self._rng_ident(20)}.")
            lines.append("\\end{document}")
            data = "\n".join(lines).encode()
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        return Sample(
            data=data,
            coarse=_COARSE["markup"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE[fmt],
        )

    # ---- data (CSV/TSV) ----
    def _gen_data(self) -> Sample:
        subtype = self.rng.choice([
            "csv", "tsv", "sql", "graphql", "protobuf", "msgpack",
            "log", "diff", "patch", "shell", "makefile", "dockerfile", "gitignore",
        ])
        if subtype in {"csv", "tsv"}:
            sep = "," if subtype == "csv" else "\t"
            cols = self.rng.randint(3, 6)
            header = sep.join(self._rng_ident(6) for _ in range(cols))
            rows = []
            for _ in range(self.rng.randint(3, 10)):
                row = sep.join(self._rng_val() for _ in range(cols))
                rows.append(row)
            data = "\n".join([header] + rows).encode()
        elif subtype == "sql":
            table = self._rng_ident(8)
            data = (
                f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, name TEXT);\n"
                f"INSERT INTO {table} VALUES (1, '{self._rng_ident(8)}');\n"
                f"SELECT id, name FROM {table} WHERE id > 0;\n"
            ).encode()
        elif subtype == "graphql":
            data = (
                "query GetItems($limit: Int!) {\n"
                f"  {self._rng_ident(8)}(limit: $limit) {{\n"
                "    id\n    name\n  }\n}\n"
            ).encode()
        elif subtype == "protobuf":
            msg = self._rng_ident(8).title()
            data = (
                'syntax = "proto3";\n'
                f"message {msg} {{\n"
                "  string id = 1;\n"
                "  int32 count = 2;\n"
                "}\n"
            ).encode()
        elif subtype == "msgpack":
            data = b"\x82\xa2id\x01\xa4name\xa5" + self._rng_ident(8).encode()
        elif subtype == "log":
            lines = []
            for _ in range(self.rng.randint(4, 10)):
                lines.append(f"2026-01-{self.rng.randint(1, 28):02d}T12:00:00Z INFO {self._rng_ident(8)}={self._rng_val()}")
            data = "\n".join(lines).encode()
        elif subtype in {"diff", "patch"}:
            data = (
                f"diff --git a/{self._rng_ident(6)}.py b/{self._rng_ident(6)}.py\n"
                "@@ -1,3 +1,3 @@\n"
                f"-old_{self._rng_ident(6)} = 1\n"
                f"+new_{self._rng_ident(6)} = 2\n"
            ).encode()
        elif subtype == "shell":
            data = (
                "#!/usr/bin/env sh\n"
                "set -eu\n"
                f"for item in {self._rng_ident(6)} {self._rng_ident(6)}; do\n"
                '  echo "$item"\n'
                "done\n"
            ).encode()
        elif subtype == "makefile":
            data = (
                ".PHONY: build test\n"
                "build:\n\tpython -m build\n"
                "test:\n\tpytest -q\n"
            ).encode()
        elif subtype == "dockerfile":
            data = (
                "FROM python:3.11-slim\n"
                "WORKDIR /app\n"
                "COPY . .\n"
                'CMD ["python", "-m", "model.pico_type.arch"]\n'
            ).encode()
        else:  # gitignore
            data = b"__pycache__/\n*.pyc\n.venv/\ndist/\nbuild/\ncheckpoints/*.pt\n"
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        return Sample(
            data=data,
            coarse=_COARSE["data"],
            modality=_MODALITY["textual"],
            subtype=_SUBTYPE.get(subtype),
        )

    # ---- link ----
    def _gen_link(self) -> Sample:
        proto = self.rng.choice(["https", "http", "ftp"])
        domain = f"{self._rng_ident(10)}.{self.rng.choice(['com', 'org', 'io', 'dev', 'ai', 'net'])}"
        path_elems = self.rng.randint(0, 4)
        path = "/" + "/".join(self._rng_ident(6) for _ in range(path_elems)) if path_elems else ""
        query = ""
        if self.rng.random() < 0.3:
            params = [f"{self._rng_ident(5)}={self._rng_ident(5)}" for _ in range(self.rng.randint(1, 3))]
            query = "?" + "&".join(params)
        data = f"{proto}://{domain}{path}{query}".encode()
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        return Sample(
            data=data,
            coarse=_COARSE["link"],
            modality=_MODALITY["textual"],
        )

    # ---- error ----
    def _gen_error(self) -> Sample:
        lang = self.rng.choice(["python", "javascript", "java", "go"])
        if lang == "python":
            lines = [
                "Traceback (most recent call last):",
                f'  File "/Users/src/{self._rng_ident(8)}.py", line {self.rng.randint(1, 100)}, in {self._rng_ident(8)}',
                f"    {self._rng_ident(12)}()",
                f'{self.rng.choice(["NameError", "TypeError", "ValueError", "KeyError", "IndexError", "AttributeError"])}: {self._rng_ident(15)}',
            ]
        elif lang == "javascript":
            lines = [
                f"Error: {self._rng_ident(15)}",
                f"    at {self._rng_ident(8)} ({self._rng_ident(8)}.js:{self.rng.randint(1, 100)}:{self.rng.randint(1, 50)})",
                f"    at {self._rng_ident(8)} ({self._rng_ident(8)}.js:{self.rng.randint(1, 100)}:{self.rng.randint(1, 50)})",
            ]
        elif lang == "java":
            lines = [
                f"Exception in thread \"main\" java.lang.{self.rng.choice(['NullPointerException', 'ArrayIndexOutOfBoundsException', 'IllegalArgumentException'])}",
                f"    at com.example.{self._rng_ident(8)}.{self._rng_ident(8)}({self._rng_ident(8)}.java:{self.rng.randint(1, 200)})",
            ]
        else:
            lines = [
                f"panic: {self._rng_ident(15)}",
                "",
                f"goroutine {self.rng.randint(1, 10)} [running]:",
                f"main.{self._rng_ident(8)}(0x{self.rng.randint(1000, 9999):x})",
            ]
        data = "\n".join(lines).encode()
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        return Sample(
            data=data,
            coarse=_COARSE["error"],
            modality=_MODALITY["textual"],
        )

    # ---- image header ----
    def _gen_image(self) -> Sample:
        img = self.rng.choice(["png", "jpeg", "gif", "webp", "bmp", "tiff"])
        if img == "png":
            header = b"\x89PNG\r\n\x1a\n"
            mime = "image/png"
        elif img == "jpeg":
            header = b"\xff\xd8\xff\xe0"
            mime = "image/jpeg"
        elif img == "gif":
            header = b"GIF89a"
            mime = "image/gif"
        elif img == "webp":
            header = b"RIFF\x00\x00\x00\x00WEBP"
            mime = "image/webp"
        elif img == "bmp":
            header = b"BM\x00\x00\x00\x00"
            mime = "image/bmp"
        else:
            header = b"II\x2a\x00"
            mime = "image/tiff"
        padding = self._rng_pad(len(header) + 60)
        data = header + padding[:(MAX_BYTES - len(header))]
        return Sample(
            data=data,
            coarse=_COARSE["image"],
            modality=_MODALITY["binary_image"],
            file_mime=_MIME[mime],
        )

    # ---- file bytes (non-image binary) ----
    def _gen_file(self) -> Sample:
        ftype = self.rng.choice([
            "pdf", "elf", "sqlite", "parquet",
            "ttf", "deb", "wasm", "bz2",
        ])
        header, mime = _BINARY_HEADERS.get(ftype, (b"\x00\x00\x00\x00", None))
        mod_map: Dict[str, str] = {
            "pdf": "binary_document",
            "elf": "binary_executable",
            "wasm": "binary_executable",
            "deb": "binary_executable",
            "bz2": "binary_other",
            "sqlite": "binary_other",
            "parquet": "binary_other",
            "ttf": "binary_other",
        }
        padding = self._rng_pad(len(header) + 60)
        data = header + padding[:(MAX_BYTES - len(header))]
        coarse = _COARSE["file"]
        modality = _MODALITY[mod_map.get(ftype, "binary_other")]
        if mime is None:
            return Sample(data=data, coarse=coarse, modality=modality)
        return Sample(data=data, coarse=coarse, modality=modality, file_mime=_MIME[mime])

    # ---- secret ----
    def _gen_secret(self) -> Sample:
        kind = self.rng.choice(["aws", "jwt", "ssh", "password", "email", "phone", "generic_api"])
        if kind == "aws":
            key = "AKIA" + "".join(self.rng.choices(string.ascii_uppercase + string.digits, k=16))
            data = f"AWS_ACCESS_KEY_ID={key}".encode()
            risk = [_RISK["api_key"]]
        elif kind == "jwt":
            h = hashlib.sha256(self.rng.randbytes(16)).digest()
            token = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}'[:-1]) + b"." + base64.urlsafe_b64encode(h)[:-1]
            data = b"Bearer " + token
            risk = [_RISK["jwt"]]
        elif kind == "ssh":
            lines = [f"-----BEGIN {self.rng.choice(['RSA', 'EC', 'OPENSSH'])} PRIVATE KEY-----"]
            for _ in range(self.rng.randint(3, 8)):
                lines.append("".join(self.rng.choices(string.ascii_letters + string.digits + "+/=", k=64)))
            lines.append("-----END PRIVATE KEY-----")
            data = "\n".join(lines).encode()
            risk = [_RISK["ssh_key"]]
        elif kind == "password":
            pwd = "".join(self.rng.choices(string.ascii_letters + string.digits + "!@#$%", k=12))
            data = f"password = \"{pwd}\"".encode()
            risk = [_RISK["password"]]
        elif kind == "email":
            addr = f"{self._rng_ident(12)}@{self._rng_ident(8)}.{self.rng.choice(['com', 'org', 'edu'])}"
            data = f"email: {addr}".encode()
            risk = [_RISK["email"]]
        elif kind == "phone":
            num = f"+1 ({self.rng.randint(200, 999)}) {self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}"
            data = f"phone: {num}".encode()
            risk = [_RISK["phone"]]
        else:
            key = self._rng_ident(12)
            val = "".join(self.rng.choices(string.ascii_letters + string.digits + "_-", k=32))
            data = f"{key}={val}".encode()
            risk = [_RISK["api_key"]]
        if len(data) > MAX_BYTES:
            data = data[:MAX_BYTES]
        if len(data) < MIN_BYTES:
            data += b"\n" + self._rng_pad(MIN_BYTES - len(data) + 10)
        return Sample(
            data=data,
            coarse=_COARSE["secret"],
            modality=_MODALITY["textual"],
            risk=risk,
        )

    # ---- archive ----
    def _gen_archive(self) -> Sample:
        arch = self.rng.choice(["zip", "tar", "gzip", "bzip2", "xz", "7z", "rar"])
        if arch == "zip":
            header = b"PK\x03\x04"
            mime = "application/zip"
        elif arch == "tar":
            header = b"ustar\x00"  # from a tar header block (256 bytes in)
            mime = "application/x-tar"
        elif arch == "gzip":
            header = b"\x1f\x8b"
            mime = "application/gzip"
        elif arch == "bzip2":
            header = b"BZh"
            mime = "application/x-bzip2"
        elif arch == "xz":
            header = b"\xfd7zXZ"
            mime = "application/x-xz"
        elif arch == "7z":
            header = b"7z\xbc\xaf\x27\x1c"
            mime = "application/x-7z-compressed"
        else:  # rar
            header = b"Rar!\x1a\x07"
            mime = "application/x-rar-compressed"
        padding = self._rng_pad(len(header) + 30)
        data = header + padding[:(MAX_BYTES - len(header))]
        return Sample(
            data=data,
            coarse=_COARSE["archive"],
            modality=_MODALITY["binary_archive"],
            file_mime=_MIME[mime],
        )

    # ---- generic binary ----
    def _gen_binary(self) -> Sample:
        length = self.rng.randint(MIN_BYTES, MAX_BYTES)
        data = bytes(self.rng.randint(0, 255) for _ in range(length))
        return Sample(
            data=data,
            coarse=_COARSE["binary"],
            modality=_MODALITY["binary_other"],
        )

    def _detect_risk(self, data: bytes) -> List[int]:
        text = data.decode("utf-8", errors="replace")
        indices: List[int] = []
        if "AKIA" in text:
            indices.append(_RISK["api_key"])
        if text.startswith("Bearer ey"):
            indices.append(_RISK["jwt"])
        if "BEGIN" in text and "PRIVATE KEY" in text:
            indices.append(_RISK["ssh_key"])
        if "password" in text.lower():
            indices.append(_RISK["password"])
        return indices


_BINARY_HEADERS: Dict[str, Tuple[bytes, str]] = {
    "pdf": (b"%PDF", "application/pdf"),
    "zip": (b"PK\x03\x04", "application/zip"),
    "gzip": (b"\x1f\x8b", "application/gzip"),
    "elf": (b"\x7fELF", "application/x-elf"),
    "sqlite": (b"SQLite format 3\x00", "application/x-sqlite3"),
    "parquet": (b"PAR1", "application/x-parquet"),
    "jar": (b"PK\x03\x04", "application/x-jar"),
    "ttf": (b"\x00\x01\x00\x00\x00", "font/ttf"),
    "deb": (b"!<arch>\ndebian-binary", "application/x-deb"),
    "wasm": (b"\x00asm", "application/wasm"),
    "ar": (b"!<arch>\n", "application/x-archive"),
    "bz2": (b"BZh", "application/x-bzip2"),
}

_TEXT_WORDS: Dict[str, List[str]] = {
    "en": ["the", "a", "an", "of", "in", "to", "and", "for", "is", "on",
           "that", "with", "this", "from", "at", "by", "be", "as", "it",
           "data", "model", "function", "return", "value", "list", "string",
           "type", "class", "file", "system", "process", "result", "error",
           "time", "number", "set", "method", "use", "way", "example"],
    "es": ["el", "la", "los", "las", "un", "una", "de", "en", "y", "que",
           "es", "por", "con", "para", "del", "como", "más", "pero", "sus",
           "entre", "datos", "función", "modelo", "valor", "lista", "tipo",
           "código", "archivo", "sistema", "resultado", "error", "tiempo"],
    "fr": ["le", "la", "les", "un", "une", "de", "en", "et", "que", "est",
           "pour", "sur", "dans", "avec", "par", "des", "sur", "plus",
           "données", "fonction", "modèle", "valeur", "liste", "type",
           "code", "fichier", "système", "résultat", "erreur", "temps"],
    "de": ["der", "die", "das", "ein", "eine", "und", "ist", "mit", "auf",
           "für", "von", "in", "zu", "aus", "bei", "als", "nach", "durch",
           "daten", "funktion", "modell", "wert", "liste", "zeichenkette",
           "code", "datei", "system", "ergebnis", "fehler", "zeit"],
    "it": ["il", "la", "lo", "un", "una", "di", "e", "che", "per",
           "con", "su", "da", "sono", "come", "ma", "più", "tra",
           "dati", "modello", "funzione", "valore", "lista", "tipo",
           "codice", "file", "sistema", "risultato", "errore", "tempo"],
    "pt": ["o", "a", "os", "as", "um", "uma", "de", "em", "e", "que",
           "para", "com", "por", "do", "da", "mais", "como", "mas",
           "dados", "função", "modelo", "valor", "lista", "tipo",
           "código", "arquivo", "sistema", "resultado", "erro", "tempo"],
    "nl": ["de", "het", "een", "van", "in", "en", "is", "met", "voor",
           "op", "aan", "dat", "door", "bij", "ook", "nog", "maar",
           "gegevens", "functie", "model", "waarde", "lijst", "type",
           "code", "bestand", "systeem", "resultaat", "fout", "tijd"],
    "sv": ["en", "ett", "av", "i", "och", "är", "på", "med", "för",
           "till", "som", "det", "den", "att", "har", "inte", "men",
           "data", "modell", "funktion", "värde", "lista", "typ",
           "kod", "fil", "system", "resultat", "fel", "tid"],
    "no": ["en", "et", "av", "i", "og", "er", "på", "med", "for",
           "til", "som", "det", "den", "har", "ikke", "men",
           "data", "modell", "funksjon", "verdi", "liste", "type",
           "kode", "fil", "system", "resultat", "feil", "tid"],
    "da": ["en", "et", "af", "i", "og", "er", "på", "med", "for",
           "til", "som", "det", "den", "har", "ikke", "men",
           "data", "model", "funktion", "værdi", "liste", "type",
           "kode", "fil", "system", "resultat", "fejl", "tid"],
    "fi": ["ja", "on", "ei", "se", "hän", "mutta", "myös", "kun",
           "tämä", "siinä", "sitten", "vielä", "aina", "mitä",
           "tiedot", "malli", "toiminto", "arvo", "lista", "tyyppi",
           "koodi", "tiedosto", "järjestelmä", "tulos", "virhe", "aika"],
    "pl": ["i", "w", "na", "z", "do", "się", "to", "jest", "jak",
           "dla", "przez", "po", "nie", "co", "ale", "tak", "od",
           "dane", "model", "funkcja", "wartość", "lista", "typ",
           "kod", "plik", "system", "wynik", "błąd", "czas"],
    "cs": ["a", "v", "na", "se", "do", "je", "to", "že", "pro",
           "za", "si", "ale", "tak", "jak", "od", "po", "při",
           "data", "model", "funkce", "hodnota", "seznam", "typ",
           "kód", "soubor", "systém", "výsledek", "chyba", "čas"],
    "sk": ["a", "v", "na", "sa", "do", "je", "to", "že", "pre",
           "za", "si", "ale", "tak", "ako", "od", "po", "pri",
           "údaje", "model", "funkcia", "hodnota", "zoznam", "typ",
           "kód", "súbor", "systém", "výsledok", "chyba", "čas"],
    "hu": ["a", "az", "és", "nem", "hogy", "is", "meg", "van",
           "egy", "csak", "már", "még", "nagyon", "minden", "között",
           "adatok", "modell", "függvény", "érték", "lista", "típus",
           "kód", "fájl", "rendszer", "eredmény", "hiba", "idő"],
    "ro": ["și", "în", "pe", "la", "cu", "din", "este", "sunt",
           "pentru", "prin", "după", "mai", "dar", "foarte",
           "date", "model", "funcție", "valoare", "listă", "tip",
           "cod", "fișier", "sistem", "rezultat", "eroare", "timp"],
    "el": ["και", "το", "σε", "από", "στο", "για", "ως", "ή",
           "δεν", "θα", "να", "τη", "με", "τα", "τον", "που",
           "δεδομένα", "μοντέλο", "συνάρτηση", "τιμή", "λίστα", "τύπος",
           "κώδικας", "αρχείο", "σύστημα", "αποτέλεσμα", "σφάλμα", "χρόνος"],
    "tr": ["ve", "bir", "bu", "ile", "için", "olarak", "daha", "çok",
           "kendi", "gibi", "kadar", "ancak", "çünkü", "sonra",
           "veri", "model", "fonksiyon", "değer", "liste", "tip",
           "kod", "dosya", "sistem", "sonuç", "hata", "zaman"],
    "ru": ["и", "в", "на", "с", "что", "как", "по", "от", "для",
           "это", "они", "но", "так", "уже", "быть", "из", "все",
           "данные", "модель", "функция", "значение", "список", "тип",
           "код", "файл", "система", "результат", "ошибка", "время"],
    "uk": ["і", "в", "на", "з", "що", "як", "по", "від", "для",
           "це", "вони", "але", "так", "вже", "бути", "до", "та",
           "дані", "модель", "функція", "значення", "список", "тип",
           "код", "файл", "система", "результат", "помилка", "час"],
    "bg": ["и", "в", "на", "с", "че", "като", "по", "от", "за",
           "това", "те", "но", "така", "вече", "бъда", "до",
           "данни", "модел", "функция", "стойност", "списък", "тип",
           "код", "файл", "система", "резултат", "грешка", "време"],
    "hr": ["i", "u", "na", "s", "da", "kako", "po", "od", "za",
           "to", "oni", "ali", "tako", "već", "biti", "do",
           "podaci", "model", "funkcija", "vrijednost", "popis", "vrsta",
           "kod", "datoteka", "sustav", "rezultat", "greška", "vrijeme"],
    "sr": ["и", "у", "на", "с", "да", "као", "по", "од", "за",
           "то", "они", "али", "тако", "већ", "бити", "до",
           "подаци", "модел", "функција", "вредност", "списак", "врста",
           "код", "датотека", "систем", "резултат", "грешка", "време"],
    "vi": ["và", "trong", "có", "của", "một", "những", "với", "được",
           "cho", "là", "các", "không", "khi", "từ", "này", "bằng",
           "dữ liệu", "mô hình", "hàm", "giá trị", "danh sách", "loại",
           "mã", "tập tin", "hệ thống", "kết quả", "lỗi", "thời gian"],
    "th": ["และ", "ใน", "มี", "ของ", "ที่", "เป็น", "จะ", "กับ",
           "ไม่", "ได้", "การ", "นี้", "ด้วย", "จาก", "ว่า",
           "ข้อมูล", "โมเดล", "ฟังก์ชัน", "ค่า", "รายการ", "ประเภท",
           "โค้ด", "ไฟล์", "ระบบ", "ผลลัพธ์", "ข้อผิดพลาด", "เวลา"],
    "id": ["dan", "di", "dari", "dengan", "untuk", "adalah", "ini",
           "itu", "tidak", "akan", "dalam", "pada", "oleh", "sebagai",
           "data", "model", "fungsi", "nilai", "daftar", "tipe",
           "kode", "berkas", "sistem", "hasil", "kesalahan", "waktu"],
    "ms": ["dan", "di", "dari", "dengan", "untuk", "adalah", "ini",
           "itu", "tidak", "akan", "dalam", "pada", "oleh", "sebagai",
           "data", "model", "fungsi", "nilai", "senarai", "jenis",
           "kod", "fail", "sistem", "hasil", "ralat", "masa"],
    "zh": ["的", "是", "在", "了", "和", "有", "不", "这", "也", "那",
           "一个", "我们", "可以", "进行", "使用", "需要", "数据", "模型",
           "函数", "变量", "列表", "类型", "代码", "文件", "系统", "错误"],
    "ja": ["の", "は", "を", "が", "に", "で", "と", "する", "ある",
           "この", "その", "データ", "モデル", "関数", "値", "リスト",
           "コード", "ファイル", "システム", "結果", "エラー", "時間",
           "ため", "こと", "もの", "よう"],
    "ko": ["의", "은", "는", "이", "가", "을", "를", "에", "에서",
           "있다", "하다", "그", "이것", "데이터", "모델", "함수",
           "값", "목록", "유형", "코드", "파일", "시스템", "결과",
           "오류", "시간", "위한"],
}


class SyntheticDataset:
    """Dataset that wraps the generator, produces N unique samples."""

    def __init__(self, generator: SyntheticGenerator, size: int):
        self.generator = generator
        self.size = size
        self._samples: Optional[List[Sample]] = None

    def __len__(self) -> int:
        return self.size

    def _ensure(self) -> List[Sample]:
        if self._samples is not None:
            return self._samples
        self._samples = [self.generator() for _ in range(self.size)]
        return self._samples

    def __getitem__(self, idx: int) -> Sample:
        return self._ensure()[idx]

    def label_counts(self) -> Dict[str, Dict[str, int]]:
        """Return class distribution for each head."""
        from collections import Counter
        samples = self._ensure()
        counts: Dict[str, Counter] = {}
        for head in (
            "coarse", "modality", "subtype",
            "code_lang", "text_lang", "file_mime",
        ):
            counter = Counter()
            for s in samples:
                v = getattr(s, head)
                if v != IGNORE_INDEX:
                    counter[v] += 1
            counts[head] = dict(counter)
        risk_counter: Dict[str, int] = {}
        for s in samples:
            for r in s.risk:
                risk_counter[RISK_LABELS[r]] = risk_counter.get(RISK_LABELS[r], 0) + 1
        counts["risk"] = risk_counter
        return {head: dict(cnt) for head, cnt in counts.items()}


def smoke_test() -> Dict[str, int]:
    gen = SyntheticGenerator(seed=42)
    ds = SyntheticDataset(gen, 500)
    print(f"Generated {len(ds)} samples")
    counts = ds.label_counts()
    total = 0
    for head, classes in counts.items():
        n = sum(classes.values())
        total += n
        print(f"  {head:10s}: {n:5d} samples across {len(classes):3d} classes")
    print(f"  {'risk':10s}: multi-label, may overlap")
    return {"samples": len(ds), "total_labels": total}


if __name__ == "__main__":
    smoke_test()
