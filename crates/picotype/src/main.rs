use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use clap::Parser;
use ndarray::Array2;
use ort::{session::Session, value::TensorRef};
use serde::Serialize;

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

#[derive(Parser)]
#[command(name = "picotype", about = "Classify content type, language, and risk")]
struct Args {
    #[arg(short = 't', long)]
    text: Option<String>,

    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    #[arg(short = 'c', long)]
    clip: bool,

    #[arg(long, default_value = "base")]
    tier: String,

    #[arg(short = 'p', long)]
    pretty: bool,
}

#[derive(Serialize)]
struct Output {
    coarse: HeadOutput,
    modality: HeadOutput,
    subtype: HeadOutput,
    code_lang: HeadOutput,
    text_lang: HeadOutput,
    file_mime: HeadOutput,
    risk: RiskOutput,
    text_length: usize,
    tier: String,
}

#[derive(Serialize)]
struct HeadOutput {
    label: String,
    confidence: f32,
}

#[derive(Serialize)]
struct RiskOutput {
    api_key: f32,
    jwt: f32,
    ssh_key: f32,
    password: f32,
    email: f32,
    phone: f32,
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn read_input(args: &Args) -> Result<String, String> {
    if let Some(text) = &args.text {
        return Ok(text.clone());
    }
    if let Some(path) = &args.file {
        return std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read file: {e}"));
    }
    if args.clip {
        let out = std::process::Command::new("pbpaste")
            .output()
            .map_err(|e| format!("Cannot read clipboard: {e}"))?;
        return String::from_utf8(out.stdout)
            .map_err(|e| format!("Clipboard not UTF-8: {e}"));
    }
    let mut buf = String::new();
    if atty::isnt(atty::Stream::Stdin) {
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("Cannot read stdin: {e}"))?;
        return Ok(buf);
    }
    Err("No input. Use --text, --file, --clip, or pipe content.".to_string())
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
    // Priority: 1. env var, 2. next to binary, 3. cwd/models, 4. cwd/checkpoints
    if let Ok(dir) = std::env::var("PICOTYPE_MODEL_DIR") {
        let p = PathBuf::from(dir).join(format!("picotype_{tier}.onnx"));
        if p.exists() { return p; }
    }

    let candidates = vec![
        std::env::current_exe().ok().and_then(|p| {
            p.parent().map(|d| d.join("../checkpoints").join(format!("picotype_{tier}.onnx")))
        }),
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

fn run(args: Args) -> Result<(), String> {
    let model_path = find_model_path(&args.tier);
    if !model_path.exists() {
        return Err(format!("ONNX model not found: {}", model_path.display()));
    }
    let mut session = Session::builder()
        .map_err(|e| format!("Cannot create ONNX session: {e}"))?
        .commit_from_file(&model_path)
        .map_err(|e| format!("Cannot load ONNX model {model_path:?}: {e}"))?;

    let text = read_input(&args)?;
    if text.trim().is_empty() {
        println!("{{\"error\": \"empty input\"}}");
        return Ok(());
    }

    let (ids, mask) = tokenize(&text);
    let ids_shape: Vec<i64> = ids.shape().iter().map(|&x| x as i64).collect();
    let mask_shape: Vec<i64> = mask.shape().iter().map(|&x| x as i64).collect();
    let outputs = session
        .run(ort::inputs![
            TensorRef::from_array_view((&ids_shape[..], ids.as_slice().unwrap())).map_err(|e| format!("Input error: {e}"))?,
            TensorRef::from_array_view((&mask_shape[..], mask.as_slice().unwrap())).map_err(|e| format!("Input error: {e}"))?,
        ])
        .map_err(|e| format!("Inference failed: {e}"))?;

    let mut out = Output {
        coarse: HeadOutput { label: String::new(), confidence: 0.0 },
        modality: HeadOutput { label: String::new(), confidence: 0.0 },
        subtype: HeadOutput { label: String::new(), confidence: 0.0 },
        code_lang: HeadOutput { label: String::new(), confidence: 0.0 },
        text_lang: HeadOutput { label: String::new(), confidence: 0.0 },
        file_mime: HeadOutput { label: String::new(), confidence: 0.0 },
        risk: RiskOutput {
            api_key: 0.0, jwt: 0.0, ssh_key: 0.0,
            password: 0.0, email: 0.0, phone: 0.0,
        },
        text_length: text.len(),
        tier: args.tier.clone(),
    };

    for (i, head) in HEAD_NAMES.iter().enumerate() {
        let tensor = outputs[i]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract {head}: {e}"))?;
        let logits: Vec<f32> = tensor.1.to_vec();

        if *head == "risk" {
            out.risk = RiskOutput {
                api_key: sigmoid(logits[0]),
                jwt: sigmoid(logits[1]),
                ssh_key: sigmoid(logits[2]),
                password: sigmoid(logits[3]),
                email: sigmoid(logits[4]),
                phone: sigmoid(logits[5]),
            };
        } else {
            let probs = softmax(&logits);
            let (best_idx, best_prob) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let label = LABELS[i][best_idx].to_string();
            match i {
                0 => out.coarse = HeadOutput { label, confidence: *best_prob },
                1 => out.modality = HeadOutput { label, confidence: *best_prob },
                2 => out.subtype = HeadOutput { label, confidence: *best_prob },
                3 => out.code_lang = HeadOutput { label, confidence: *best_prob },
                4 => out.text_lang = HeadOutput { label, confidence: *best_prob },
                5 => out.file_mime = HeadOutput { label, confidence: *best_prob },
                _ => {}
            }
        }
    }

    let json = if args.pretty {
        serde_json::to_string_pretty(&out)
    } else {
        serde_json::to_string(&out)
    }
    .unwrap_or_else(|_| "{}".to_string());
    println!("{json}");
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
