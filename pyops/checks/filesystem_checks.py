from pathlib import Path

def _result(name, ok, msg):
    return {"dataset": "repo", "check": name, "ok": ok, "message": msg}

def check_required_paths(root: Path, required: list):
    results = []
    for rel in required:
        p = (root / rel)
        ok = p.exists()
        results.append(_result(f"required::{rel}", ok, f"{rel} {'exists' if ok else 'MISSING'}"))
    return results

def check_dockerfile_basics(dockerfile: Path):
    name = "dockerfile::basic"
    if not dockerfile.exists():
        return [_result(name, False, "Dockerfile MISSING")]
    text = dockerfile.read_text(encoding="utf-8", errors="ignore")

    msgs = []
    ok = True

    if "FROM" not in text:
        ok = False
        msgs.append("No FROM found")
    if "FROM" in text and ":latest" in text:
        ok = False
        msgs.append("Avoid FROM ...:latest")
    if "pip install -r requirements.txt" not in text and "poetry install" not in text:
        msgs.append("Hint: no install line found (pip/poetry)")
    return [_result(name, ok, "; ".join(msgs) if msgs else "Dockerfile basic OK")]
