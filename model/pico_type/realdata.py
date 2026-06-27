"""Fetch real code samples from GitHub for training."""

from __future__ import annotations

import json
import random
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

from .data import MAX_BYTES, MIN_BYTES, Sample
from .labels import COARSE_LABELS, MODALITY_LABELS, CODE_LANG_LABELS

_COARSE = {name: i for i, name in enumerate(COARSE_LABELS)}
_MODALITY = {name: i for i, name in enumerate(MODALITY_LABELS)}
_CODE = {name: i for i, name in enumerate(CODE_LANG_LABELS)}

# Map CODE_LANG_LABELS names to GitHub API language names
GITHUB_LANG_MAP: Dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    "java": "java",
    "kotlin": "kotlin",
    "scala": "scala",
    "groovy": "groovy",
    "clojure": "clojure",
    "c": "c",
    "cpp": "c++",
    "csharp": "c#",
    "fsharp": "f#",
    "objectivec": "objective-c",
    "go": "go",
    "rust": "rust",
    "zig": "zig",
    "ruby": "ruby",
    "php": "php",
    "perl": "perl",
    "lua": "lua",
    "tcl": "tcl",
    "swift": "swift",
    "dart": "dart",
    "julia": "julia",
    "nim": "nim",
    "crystal": "crystal",
    "haskell": "haskell",
    "ocaml": "ocaml",
    "elm": "elm",
    "erlang": "erlang",
    "elixir": "elixir",
    "lisp": "common-lisp",
    "scheme": "scheme",
    "racket": "racket",
    "r": "r",
    "matlab": "matlab",
    "octave": "matlab",
    "sas": "sas",
    "stata": "stata",
    "html": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "bash": "shell",
    "zsh": "shell",
    "fish": "shell",
    "powershell": "powershell",
    "vim": "vim-script",
    "fortran": "fortran",
    "cobol": "cobol",
    "ada": "ada",
    "pascal": "pascal",
    "delphi": "pascal",
    "vb": "visual-basic",
    "prolog": "prolog",
    "vhdl": "vhdl",
    "plsql": "sql",
    "tsql": "sql",
}

# Known repos with permissive licenses for each language
FALLBACK_REPOS: Dict[str, List[str]] = {
    "python": [
        "python/cpython", "django/django", "pallets/flask",
        "psf/requests", "pypa/pip", "numpy/numpy", "scipy/scipy",
        "matplotlib/matplotlib", "pandas-dev/pandas", "scikit-learn/scikit-learn",
        "pytorch/pytorch", "tensorflow/tensorflow", "ansible/ansible",
        "home-assistant/core", "mitmproxy/mitmproxy", "httpie/cli",
        "sqlalchemy/sqlalchemy", "pytest-dev/pytest", "mitsuhiko/flask",
        "celery/celery", "apache/airflow", "apache/superset",
    ],
    "javascript": [
        "expressjs/express", "lodash/lodash", "nodejs/node",
        "axios/axios", "chartjs/Chart.js", "moment/moment",
        "webpack/webpack", "sveltejs/svelte", "preactjs/preact",
        "babel/babel", "eslint/eslint", "jquery/jquery",
        "semantic-release/semantic-release", "mochajs/mocha",
        "videojs/video.js", "jhipster/generator-jhipster",
    ],
    "typescript": [
        "microsoft/typescript", "microsoft/vscode",
        "nestjs/nest", "typeorm/typeorm", "prisma/prisma",
        "nestjs/nest", "excalidraw/excalidraw", "calcom/cal.com",
        "n8n-io/n8n", "appwrite/appwrite", "triggerdev/trigger.dev",
    ],
    "jsx": ["facebook/react", "vercel/next.js", "gatsbyjs/gatsby"],
    "tsx": ["facebook/react", "vercel/next.js", "remix-run/react-router"],
    "java": [
        "spring-projects/spring-boot", "elastic/elasticsearch",
        "apache/hadoop", "apache/dubbo", "netty/netty",
        "google/guava", "apache/tomcat", "apache/maven",
        "eclipse/jetty.project", "hibernate/hibernate-orm",
        "apache/kafka", "apache/zookeeper",
        "jenkinsci/jenkins", "keycloak/keycloak",
    ],
    "kotlin": [
        "JetBrains/kotlin", "square/okhttp",
        "Kotlin/kotlinx.coroutines", "Kotlin/kotlinx.serialization",
        "mockk/mockk", "ktorio/ktor", "JetBrains/compose-multiplatform",
    ],
    "scala": ["scala/scala", "apache/spark", "twitter/finagle", "akka/akka"],
    "groovy": ["apache/groovy", "gradle/gradle", "grails/grails-core"],
    "clojure": ["clojure/clojure", "technomancy/leiningen", "noprompt/garden"],
    "c": [
        "torvalds/linux", "redis/redis", "git/git",
        "libuv/libuv", "curl/curl", "nginx/nginx",
        "sqlite/sqlite", "openssl/openssl", "FFmpeg/FFmpeg",
        "tmux/tmux", "vim/vim", "libevent/libevent",
        "stedolan/jq", "memcached/memcached",
    ],
    "cpp": [
        "nlohmann/json", "ocornut/imgui", "electron/electron",
        "google/googletest", "google/leveldb", "facebook/folly",
        "tensorflow/tensorflow", "apache/arrow", "fmtlib/fmt",
        "microsoft/calculator", "zealdocs/zeal", "yhirose/cpp-httplib",
    ],
    "csharp": [
        "dotnet/runtime", "dotnet/aspnetcore", "2dust/v2rayN",
        "JamesNK/Newtonsoft.Json", "AutoMapper/AutoMapper",
        "fluentmigrator/fluentmigrator", "serilog/serilog",
    ],
    "fsharp": ["dotnet/fsharp", "fsprojects/FSharpPlus", "fsprojects/Argu", "fscheck/FsCheck", "fslaborg/FSharpLab.io"],
    "objectivec": [
        "AFNetworking/AFNetworking", "SDWebImage/SDWebImage",
        "realm/realm-swift", "CocoaLumberjack/CocoaLumberjack",
    ],
    "go": [
        "golang/go", "kubernetes/kubernetes",
        "gin-gonic/gin", "gohugoio/hugo", "moby/moby",
        "prometheus/prometheus", "etcd-io/etcd", "traefik/traefik",
        "hashicorp/terraform", "hashicorp/vault", "minio/minio",
        "grafana/k6", "grpc/grpc-go", "helm/helm",
        "coredns/coredns", "containerd/containerd",
    ],
    "rust": [
        "rust-lang/rust", "tokio-rs/tokio",
        "serde-rs/serde", "actix/actix-web", "denoland/deno",
        "ruffle-rs/ruffle", "swc-project/swc", "neovide/neovide",
        "meilisearch/meilisearch", "ajeetdsouza/zoxide",
        "sharkdp/bat", "BurntSushi/ripgrep",
        "Wilfred/difftastic", "dandavison/delta",
    ],
    "zig": ["ziglang/zig", "ziglang/zig-spec"],
    "ruby": [
        "rails/rails", "jekyll/jekyll",
        "ruby/ruby", "Homebrew/brew", "chatwoot/chatwoot",
        "mastodon/mastodon", "discourse/discourse",
        "heartcombo/devise", "dependabot/dependabot-core",
    ],
    "php": [
        "laravel/laravel", "symfony/symfony",
        "composer/composer", "pestphp/pest", "phpstan/phpstan",
        "coollabsio/coolify", "filamentphp/filament",
        "magento/magento2", "woocommerce/woocommerce",
    ],
    "perl": ["AlDanial/cloc", "duckduckgo/duckduckgo", "Perl/perl5"],
    "lua": ["lua/lua", "neovim/neovim", "LuaLS/lua-language-server"],
    "tcl": ["tcltk/tcl", "flightaware/tcllib"],
    "swift": [
        "apple/swift", "Alamofire/Alamofire",
        "vapor/vapor", "pointfreeco/swift-composable-architecture",
        "airbnb/lottie-ios", "swiftlang/swift-package-manager",
    ],
    "dart": ["dart-lang/sdk", "flutter/flutter", "dart-lang/package_config"],
    "julia": ["JuliaLang/julia", "JuliaDiffEq/DifferentialEquations.jl"],
    "nim": ["nim-lang/Nim"],
    "crystal": ["crystal-lang/crystal"],
    "haskell": ["koalaman/shellcheck", "hadolint/hadolint", "jgm/pandoc"],
    "ocaml": ["ocaml/ocaml", "coq/coq", "semgrep/semgrep"],
    "elm": ["elm/compiler"],
    "erlang": ["erlang/otp", "ninenines/cowboy", "emqx/emqx"],
    "elixir": ["elixir-lang/elixir", "phoenixframework/phoenix", "plausible/analytics"],
    "lisp": ["sbcl/sbcl", "fukamachi/ningle", "robert-strandh/Cluffer"],
    "scheme": ["racket/racket", "arcfide/sagittarius-scheme"],
    "racket": ["racket/racket", "racket/typed-racket"],
    "r": ["tidyverse/ggplot2", "tidyverse/dplyr", "r-lib/testthat"],
    "matlab": ["TianhongDai/integrated-human-model"],
    "octave": ["gnu-octave/octave"],
    "sas": ["sassoftware/sas-macros"],
    "stata": ["gslab-econ/ra-gslab"],
    "html": ["facebook/react", "whatwg/html", "google/material-design-lite"],
    "css": ["tailwindlabs/tailwindcss", "necolas/normalize.css", "animate-css/animate.css"],
    "scss": ["twbs/bootstrap", "primer/css"],
    "sass": ["sass/sass"],
    "less": ["less/less.js"],
    "bash": ["xonsh/xonsh", "scop/bash-completion", "dylanaraps/pure-bash-bible"],
    "zsh": ["ohmyzsh/ohmyzsh", "zsh-users/zsh-autosuggestions", "powerlevel10k/powerlevel10k"],
    "fish": ["fish-shell/fish-shell"],
    "powershell": ["PowerShell/PowerShell", "MicrosoftDocs/azure-docs-powershell"],
    "vim": ["vim/vim", "neovim/neovim", "SpaceVim/SpaceVim", "vim-airline/vim-airline", "junegunn/fzf"],
    "fortran": ["Reference-LAPACK/lapack", "fortran-lang/stdlib", "fortran-lang/fpm", "certik/fortran-compiler"],
    "cobol": ["opensourcecobol/opensource-cobol", "COBOL-Stack/COBOL-Stack"],
    "ada": ["AdaCore/gnatcoll-core", "AdaCore/aws", "AdaCore/gtkada"],
    "pascal": ["cheahengsoon/ZeosLib", "pascal-network/lazarus", "fpc/fpc"],
    "delphi": ["cheahengsoon/ZeosLib", "pascal-network/lazarus"],
    "vb": ["dotnet/vblang", "dotnet/roslyn", "dotnet/runtime"],
    "prolog": ["SWI-Prolog/swipl-devel", "triska/clpfd", "infradig/inclpr"],
    "vhdl": ["ghdl/ghdl", "VHDL-LS/rust_hdl", "stnolting/neorv32"],
    "sql": ["apache/hive", "ClickHouse/ClickHouse"],
    "plsql": ["oracle/db-sample-schemas"],
    "tsql": ["microsoft/mssql-server-samples"],
}

# File extensions to filter by language
LANG_EXTENSIONS: Dict[str, List[str]] = {
    "python": [".py"],
    "javascript": [".js", ".mjs"],
    "typescript": [".ts"],
    "jsx": [".jsx"],
    "tsx": [".tsx"],
    "java": [".java"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "groovy": [".groovy", ".gvy"],
    "clojure": [".clj", ".cljs", ".cljc"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".cxx"],
    "csharp": [".cs"],
    "fsharp": [".fs", ".fsx"],
    "objectivec": [".m", ".mm"],
    "go": [".go"],
    "rust": [".rs"],
    "zig": [".zig"],
    "ruby": [".rb"],
    "php": [".php"],
    "perl": [".pl", ".pm"],
    "lua": [".lua"],
    "tcl": [".tcl"],
    "swift": [".swift"],
    "dart": [".dart"],
    "julia": [".jl"],
    "nim": [".nim"],
    "crystal": [".cr"],
    "haskell": [".hs"],
    "ocaml": [".ml", ".mli"],
    "elm": [".elm"],
    "erlang": [".erl", ".hrl"],
    "elixir": [".ex", ".exs"],
    "lisp": [".lisp", ".cl", ".el"],
    "scheme": [".scm", ".ss"],
    "racket": [".rkt"],
    "r": [".r", ".R"],
    "matlab": [".m"],
    "octave": [".m"],
    "sas": [".sas"],
    "stata": [".do", ".ado"],
    "html": [".html", ".htm"],
    "css": [".css"],
    "scss": [".scss"],
    "sass": [".sass"],
    "less": [".less"],
    "bash": [".sh"],
    "zsh": [".zsh"],
    "fish": [".fish"],
    "powershell": [".ps1", ".psm1"],
    "vim": [".vim"],
    "fortran": [".f", ".f90", ".f95", ".f03"],
    "cobol": [".cbl", ".cob"],
    "ada": [".ada", ".adb", ".ads"],
    "pascal": [".pas"],
    "delphi": [".pas", ".dpr"],
    "vb": [".vb", ".bas"],
    "prolog": [".pl", ".pro"],
    "vhdl": [".vhd", ".vhdl"],
    "sql": [".sql"],
    "plsql": [".sql", ".pks", ".pkb"],
    "tsql": [".sql"],
}


def github_search_code(
    language: str,
    token: str = "",
    per_page: int = 30,
    max_results: int = 100,
) -> List[Dict]:
    """Search GitHub for code files in a given language."""
    gh_lang = GITHUB_LANG_MAP.get(language, language)
    results: List[Dict] = []
    page = 1

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "pico-type/1.0",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    while len(results) < max_results and page <= 10:
        url = (
            f"https://api.github.com/search/code"
            f"?q=language:{gh_lang}&per_page={min(per_page, max_results - len(results))}"
            f"&page={page}&sort=indexed&order=desc"
        )
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                items = data.get("items", [])
                if not items:
                    break
                results.extend(items)
                page += 1
                # Respect rate limiting
                rem = int(resp.headers.get("X-RateLimit-Remaining", 0))
                if rem < 5:
                    time.sleep(60)
                else:
                    time.sleep(0.5)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"  Rate limited for {language}. Waiting 60s...")
                time.sleep(60)
                continue
            elif e.code == 422:
                print(f"  Search API not supported for {language} ({gh_lang})")
                break
            else:
                print(f"  HTTP {e.code} for {language}: {e}")
                break
        except Exception as e:
            print(f"  Error searching {language}: {e}")
            break
    return results


def fetch_raw(url: str, token: str = "") -> Optional[bytes]:
    """Download raw content from a URL."""
    headers = {"User-Agent": "pico-type/1.0"}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()
    except Exception:
        return None


def search_download_code(
    language: str,
    samples_per_lang: int = 10,
    token: str = "",
) -> List[bytes]:
    """Search GitHub and download raw code files for a language."""
    results = github_search_code(language, token, per_page=30, max_results=samples_per_lang * 2)
    downloaded: List[bytes] = []
    seen_content: set = set()

    for item in results:
        if len(downloaded) >= samples_per_lang:
            break
        raw_url = item.get("html_url", "").replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        if not raw_url:
            continue
        content = fetch_raw(raw_url, token)
        if content is None or len(content) < MIN_BYTES or len(content) > MAX_BYTES:
            continue
        # Deduplicate
        content_hash = hash(content[:100])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        downloaded.append(content)

    return downloaded


def fetch_from_fallback_repo(
    repo: str,
    extension: str,
    token: str = "",
    max_files: int = 15,
) -> List[bytes]:
    """Fetch files from a known GitHub repo by extension."""
    files: List[bytes] = []
    api_url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "pico-type/1.0"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            tree = json.loads(resp.read()).get("tree", [])
    except Exception as e:
        print(f"  Error fetching repo {repo}: {e}")
        return files

    ext = extension if extension.startswith(".") else f".{extension}"
    candidates = [
        item for item in tree
        if item.get("path", "").endswith(ext) and item["type"] == "blob"
        and not any(p in item.get("path", "") for p in ["test/", "tests/", "spec/", "doc/", "docs/", "example/", "examples/", "benchmark/", "benchmarks/"])
    ]
    random.shuffle(candidates)

    for item in candidates[:max_files * 2]:
        if len(files) >= max_files:
            break
        raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{item['path']}"
        content = fetch_raw(raw_url, token)
        if content and MIN_BYTES <= len(content) <= MAX_BYTES:
            files.append(content)

    return files


def get_real_code_samples(
    language: str,
    samples_per_lang: int = 30,
    token: str = "",
) -> List[Sample]:
    """Get real code Samples for a given language from GitHub."""
    lang_idx = _CODE.get(language)
    if lang_idx is None:
        return []

    all_content: List[bytes] = []
    ext = LANG_EXTENSIONS.get(language, [".txt"])[0]

    # Strategy 1: Search API
    downloaded = search_download_code(language, samples_per_lang, token)
    all_content.extend(downloaded)
    print(f"  Search API: {len(downloaded)}/{samples_per_lang} for {language}")

    # Strategy 2: Fallback repos (much larger list now)
    if len(all_content) < samples_per_lang:
        repos = FALLBACK_REPOS.get(language, [])
        random.shuffle(repos)
        remaining = samples_per_lang - len(all_content)
        per_repo = max(3, remaining // max(1, len(repos)))
        for repo in repos:
            files = fetch_from_fallback_repo(repo, ext, token, per_repo)
            all_content.extend(files)
            if len(all_content) >= samples_per_lang:
                break

    # Truncate and deduplicate by content hash
    seen = set()
    unique = []
    for c in all_content:
        h = hash(c[:200])
        if h not in seen:
            seen.add(h)
            unique.append(c)
    all_content = unique[:samples_per_lang]

    samples = []
    for content in all_content:
        data = content[:MAX_BYTES]
        samples.append(Sample(
            data=data,
            coarse=_COARSE["code"],
            modality=_MODALITY["textual"],
            code_lang=lang_idx,
        ))

    return samples


def build_real_code_dataset(
    languages: Optional[List[str]] = None,
    samples_per_lang: int = 10,
    token: str = "",
) -> Tuple[List[Sample], Dict[str, int]]:
    """Build a dataset of real code samples from GitHub."""
    if languages is None:
        languages = CODE_LANG_LABELS

    all_samples: List[Sample] = []
    stats: Dict[str, int] = {}

    for lang in languages:
        samples = get_real_code_samples(lang, samples_per_lang, token)
        all_samples.extend(samples)
        stats[lang] = len(samples)
        print(f"  {lang:12s}: {len(samples)} samples")
        time.sleep(0.2)  # Be nice to GitHub

    random.shuffle(all_samples)
    return all_samples, stats


class MixedDataset:
    """Dataset that mixes real code samples with synthetic samples."""

    def __init__(
        self,
        real_code_samples: List[Sample],
        synthetic_generator,
        synthetic_ratio: float = 0.5,
        total_size: int = 10000,
    ):
        self.real_code = real_code_samples
        self.gen = synthetic_generator
        self.synthetic_ratio = synthetic_ratio
        self.total_size = total_size
        self.real_per_epoch = len(real_code_samples)

    def get_batch_mix(self, batch_size: int) -> Tuple[List[Sample], int]:
        """Return a mixed batch: (samples, num_real_in_batch)."""
        n_real = min(int(batch_size * self.synthetic_ratio), self.real_per_epoch)
        n_synth = batch_size - n_real

        batch_real = random.sample(self.real_code, n_real) if n_real > 0 and self.real_code else []
        batch_synth = [self.gen() for _ in range(n_synth)]

        combined = batch_real + batch_synth
        random.shuffle(combined)
        return combined, len(batch_real)

    def all_samples(self) -> List[Sample]:
        synth = [self.gen() for _ in range(self.total_size - len(self.real_code))]
        combined = self.real_code + synth
        random.shuffle(combined)
        return combined
