use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use ndarray::Array2;
use ort::{session::Session, value::TensorRef};


const MAX_BYTES: usize = 1024;

const LABELS: &[&[&str]] = &[
    &[
        "text", "code", "link", "image", "file", "config", "markup", "data",
        "error", "secret", "archive", "binary",
    ],
    &[
        "textual", "binary_image", "binary_archive", "binary_executable",
        "binary_document", "binary_audio", "binary_video", "binary_other",
    ],
    &[
        "json", "yaml", "toml", "ini", "csv", "tsv", "xml", "html", "markdown",
        "rst", "asciidoc", "tex", "sql", "graphql", "protobuf", "msgpack",
        "log", "diff", "patch", "env", "shell", "makefile", "dockerfile", "gitignore",
    ],
    &[
        "python", "javascript", "typescript", "jsx", "tsx", "java", "kotlin",
        "scala", "groovy", "clojure", "c", "cpp", "csharp", "fsharp", "objectivec",
        "go", "rust", "zig", "ruby", "php", "perl", "lua", "tcl", "swift", "dart",
        "julia", "nim", "crystal", "haskell", "ocaml", "elm", "erlang", "elixir",
        "lisp", "scheme", "racket", "r", "matlab", "octave", "sas", "stata",
        "sql", "plsql", "tsql", "html", "css", "scss", "sass", "less",
        "bash", "zsh", "fish", "powershell", "vim", "fortran", "cobol", "ada",
        "pascal", "delphi", "vb", "prolog", "vhdl",
    ],
    &[
        "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da", "fi", "pl",
        "cs", "sk", "hu", "ro", "el", "tr", "ru", "uk", "bg", "sr", "hr",
        "zh", "ja", "ko", "ar", "hi", "th", "vi",
    ],
    &[
        "text/html", "application/json", "application/xml", "text/yaml",
        "text/toml", "text/ini", "text/csv", "text/tsv", "text/markdown",
        "text/plain", "text/x-python", "text/x-java", "text/x-c",
        "text/x-cpp", "text/x-rust", "text/x-go", "text/x-ruby",
        "text/x-php", "text/x-javascript", "text/x-typescript",
        "text/x-shellscript", "text/x-sql", "text/x-dockerfile",
        "text/x-makefile", "text/x-yaml", "text/x-diff", "text/x-log",
        "text/x-env", "text/x-tex", "text/x-asciidoc", "text/x-rst",
        "application/pdf", "application/zip", "application/gzip",
        "application/x-tar", "application/x-7z-compressed", "application/x-rar-compressed",
        "application/x-bzip2", "application/x-xz", "application/x-iso9660-image",
        "application/vnd.sqlite3", "application/x-parquet",
        "application/x-elf", "application/x-mach-binary",
        "application/x-pe-executable", "application/java-archive",
        "application/wasm", "application/vnd.debian.binary-package",
        "application/x-apple-diskimage", "application/x-msdownload",
        "application/x-sharedlib", "application/x-object",
        "application/x-pcap", "application/x-hdf5", "application/x-netcdf",
        "application/xml", "application/atom+xml", "application/rss+xml",
        "application/rdf+xml", "application/xhtml+xml",
        "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp",
        "image/tiff", "image/svg+xml", "image/x-icon", "image/avif",
        "audio/mpeg", "audio/wav", "audio/ogg", "audio/flac",
        "audio/aac", "audio/mp4", "audio/webm",
        "video/mp4", "video/webm", "video/ogg", "video/x-msvideo",
        "video/quicktime", "video/x-matroska",
        "font/ttf", "font/otf", "font/woff", "font/woff2",
        "application/octet-stream", "application/unknown",
    ],
];

const HEAD_NAMES: &[&str] = &[
    "coarse", "modality", "subtype", "code_lang", "text_lang", "file_mime", "risk",
];

const RISK_LABELS: &[&str] = &[
    "api_key", "jwt", "ssh_key", "password", "email", "phone",
];

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tokenize(text: &str) -> (Array2<i64>, Array2<bool>) {
    let bytes = text.as_bytes();
    let len = bytes.len().min(MAX_BYTES);
    let mut ids = Array2::zeros((1, MAX_BYTES));
    let mut mask = Array2::from_elem((1, MAX_BYTES), false);

    for i in 0..len {
        ids[[0, i]] = bytes[i] as i64;
        mask[[0, i]] = true;
    }
    (ids, mask)
}

fn find_model_path(tier: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("PICOTYPE_MODEL_DIR") {
        let p = PathBuf::from(dir).join(format!("picotype_{tier}.onnx"));
        if p.exists() {
            return p;
        }
    }

    let candidates = vec![
        Some(PathBuf::from("models").join(format!("picotype_{tier}.onnx"))),
        Some(PathBuf::from("checkpoints").join(format!("picotype_{tier}.onnx"))),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            return candidate;
        }
    }

    PathBuf::from("models").join(format!("picotype_{tier}.onnx"))
}

fn load_session(tier: &str) -> Result<Session, String> {
    let model_path = find_model_path(tier);
    Session::builder()
        .map_err(|e| format!("Cannot create ONNX session: {e}"))?
        .commit_from_file(&model_path)
        .map_err(|e| format!("Cannot load ONNX model {model_path:?}: {e}"))
}

fn classify(session: &mut Session, text: &str, tier: &str) -> Result<serde_json::Value, String> {
    if text.trim().is_empty() {
        return Ok(serde_json::json!({"error": "empty input"}));
    }

    let (ids, mask) = tokenize(text);
    let ids_shape: Vec<i64> = ids.shape().iter().map(|&x| x as i64).collect();
    let mask_shape: Vec<i64> = mask.shape().iter().map(|&x| x as i64).collect();

    let outputs = session
        .run(ort::inputs![
            TensorRef::from_array_view((&ids_shape[..], ids.as_slice().unwrap()))
                .map_err(|e| format!("Input error: {e}"))?,
            TensorRef::from_array_view((&mask_shape[..], mask.as_slice().unwrap()))
                .map_err(|e| format!("Input error: {e}"))?,
        ])
        .map_err(|e| format!("Inference failed: {e}"))?;

    let mut result = serde_json::Map::new();

    for (i, head) in HEAD_NAMES.iter().enumerate() {
        let tensor = outputs[i]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract {head}: {e}"))?;
        let logits: Vec<f32> = tensor.1.to_vec();

        if *head == "risk" {
            let risk_flags: Vec<serde_json::Value> = RISK_LABELS
                .iter()
                .enumerate()
                .filter_map(|(j, label)| {
                    let prob = sigmoid(logits[j]);
                    if prob > 0.3 {
                        Some(serde_json::json!({"label": label, "confidence": (prob * 10000.0).round() / 10000.0}))
                    } else {
                        None
                    }
                })
                .collect();
            let risk_scores: serde_json::Map<String, serde_json::Value> = RISK_LABELS
                .iter()
                .enumerate()
                .map(|(j, label)| (label.to_string(), serde_json::json!(sigmoid(logits[j]))))
                .collect();
            result.insert("risk_flags".to_string(), serde_json::json!(risk_flags));
            result.insert("risk_scores".to_string(), serde_json::json!(risk_scores));
        } else {
            let probs = softmax(&logits);
            let (best_idx, best_prob) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let label = LABELS[i][best_idx].to_string();
            let head_key = match *head {
                "code_lang" => "code_language",
                "text_lang" => "text_language",
                "file_mime" => "file_mime",
                _ => head,
            };
            let mut head_map = serde_json::Map::new();
            head_map.insert("label".to_string(), serde_json::json!(label));
            head_map.insert("confidence".to_string(), serde_json::json!((best_prob * 10000.0).round() / 10000.0));
            result.insert(head_key.to_string(), serde_json::json!(head_map));
        }
    }

    result.insert("text_length".to_string(), serde_json::json!(text.len()));
    result.insert("model_tier".to_string(), serde_json::json!(tier));

    Ok(serde_json::Value::Object(result))
}

fn handle_request(req: &serde_json::Value, session: &mut Session) -> serde_json::Value {
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let req_id = req.get("id");

    match method {
        "initialize" => {
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {
                            "classify": {
                                "description": "Classify content type, language, modality, and risk",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string", "description": "Content to classify"},
                                        "tier": {"type": "string", "enum": ["tiny", "small", "base", "pro"], "default": "base"}
                                    },
                                    "required": ["text"]
                                }
                            },
                            "classify_file": {
                                "description": "Classify a file's content type and risk",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "description": "File path"},
                                        "tier": {"type": "string", "enum": ["tiny", "small", "base", "pro"], "default": "base"}
                                    },
                                    "required": ["path"]
                                }
                            }
                        }
                    }
                }
            })
        }
        "tools/call" => {
            let params = req.get("params");
            let tool = params.and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
            let args = params.and_then(|p| p.get("arguments")).cloned().unwrap_or(serde_json::Value::Null);

            match tool {
                "classify" => {
                    let text = args.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    let tier = args.get("tier").and_then(|t| t.as_str()).unwrap_or("base");
                    match classify(session, text, tier) {
                        Ok(result) => {
                            serde_json::json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "content": [{"type": "json", "json": result}]
                                }
                            })
                        }
                        Err(e) => {
                            serde_json::json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32000, "message": e}
                            })
                        }
                    }
                }
                "classify_file" => {
                    let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                    let tier = args.get("tier").and_then(|t| t.as_str()).unwrap_or("base");
                    match std::fs::read_to_string(path) {
                        Ok(text) => match classify(session, &text, tier) {
                            Ok(result) => {
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "result": {
                                        "content": [{"type": "json", "json": result}]
                                    }
                                })
                            }
                            Err(e) => {
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {"code": -32000, "message": e}
                                })
                            }
                        },
                        Err(_) => {
                            serde_json::json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32000, "message": format!("File not found: {path}")}
                            })
                        }
                    }
                }
                _ => {
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": format!("Unknown tool: {tool}")}
                    })
                }
            }
        }
        _ => {
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": format!("Unknown method: {method}")}
            })
        }
    }
}

fn main() {
    let tier = std::env::var("PICOTYPE_TIER").unwrap_or_else(|_| "base".to_string());

    let mut session = match load_session(&tier) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let reader = stdin.lock();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let req: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err = serde_json::json!({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": format!("Parse error: {e}")}
                });
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let _ = writeln!(handle, "{}", err);
                continue;
            }
        };

        let resp = handle_request(&req, &mut session);
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        let _ = writeln!(handle, "{}", resp);
    }
}
