import sys
from pathlib import Path
from checks.filesystem_checks import check_required_paths, check_dockerfile_basics
from checks.secret_scanner import scan_for_secrets
from report.junit import to_junit_xml

def main(out_dir="reports"):
    repo_root = Path(__file__).resolve().parents[1]

    results = []
    # 1) Required files and directories
    required = [
        "src",
        "src/cargar_datos.py",
        "src/model_training.py",
        "src/model_evaluation.py",
        "src/model_deploy.py",
        "Dockerfile",
        "README.md",
        "pyproject.toml",   # Poetry
        #"jenkins",          #  Jenkins
    ]
    results += check_required_paths(repo_root, required)

    # 2) Dockerfile
    results += check_dockerfile_basics(repo_root / "Dockerfile")

    # 3) Secrets scanning
    #    (.git, venv, lockfiles and pkl/db for performance)
    ignore_globs = [
        ".git/**", ".venv/**", "venv/**", "**/__pycache__/**",
        "**/*.pkl", "**/*.db", "models/**", "monitoring.db"
    ]
    results += scan_for_secrets(repo_root, ignore_globs)

    # --- Export JUnit XML and exit code ---
    out_path = repo_root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    xml = to_junit_xml(results, "pyops_checks")
    (out_path / "junit.xml").write_bytes(xml)

    fails = [r for r in results if not r["ok"]]
    print(f"Checks: {len(results)} | Fails: {len(fails)}")
    for r in fails[:30]:
        print(f"[FAIL] {r['check']} -> {r['message']}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports")
