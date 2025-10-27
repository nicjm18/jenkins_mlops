import re
from pathlib import Path
from fnmatch import fnmatch

_PATTERNS = [
    # AWS
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}", "AWS Secret Access Key"),
    # GitHub
    (r"ghp_[A-Za-z0-9]{36,}", "GitHub PAT"),
    # Google
    (r"AIza[0-9A-Za-z\-_]{35}", "Google API Key"),
    # Azure (SAS/keys) – genérico
    (r"(?i)(AccountKey|SharedAccessSignature|sas_token)\s*=\s*[A-Za-z0-9%+/=_.\-]{20,}", "Azure key/SAS"),
    # Generic tokens
    (r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", "Generic secret literal"),
]

_TEXT_EXT = {
    ".py", ".json", ".yml", ".yaml", ".toml", ".md", ".txt",
    ".env", ".cfg", ".ini", ".dockerignore", ".gitignore", ".sh"
}

def _result(path, pattern_name, ok, msg):
    return {"dataset": str(path), "check": f"secrets::{pattern_name}", "ok": ok, "message": msg}

def _should_ignore(rel_path: str, ignore_globs: list):
    return any(fnmatch(rel_path, g) for g in ignore_globs)

def scan_for_secrets(repo_root: Path, ignore_globs: list):
    results = []
    for p in repo_root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(repo_root).as_posix()
        if _should_ignore(rel, ignore_globs):
            continue
        if p.suffix.lower() not in _TEXT_EXT:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern, label in _PATTERNS:
            if re.search(pattern, text):
                results.append(_result(rel, label, False, f"Possible secret found in {rel} ({label})"))
    # Si no se encontraron coincidencias, reporta un OK global
    if not any(not r["ok"] for r in results):
        results.append({"dataset": "repo", "check": "secrets::global", "ok": True, "message": "No obvious secrets found"})
    return results
