"""Build a private-source-free copy and compare all generated reader bytes."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    issues = []
    with tempfile.TemporaryDirectory(prefix="source-free-", dir=dist) as temporary:
        destination = Path(temporary)
        directories = ["assets", "chapters", "frontmatter", "labs", "tools", "plan"]
        for name in directories:
            shutil.copytree(ROOT / name, destination / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.part"))
        for name in ["source-manifest.json", "completion.json", "requirements.txt",
                     "requirements-lock.txt", "code_lab.py", "README.md", "WORKFLOW.md",
                     ".nojekyll"]:
            shutil.copy2(ROOT / name, destination / name)
        for name in ["validation/chapters", "sources", "reviews"]:
            (destination / name).mkdir(parents=True, exist_ok=True)
        for path in (ROOT / "validation/chapters").glob("*.json"):
            shutil.copy2(path, destination / "validation/chapters" / path.name)
        for path in (ROOT / "validation").glob("*.json"):
            if path.name != "source-free-checks.json":
                shutil.copy2(path, destination / "validation" / path.name)
        for path in (ROOT / "reviews").glob("ch*-r*.json"):
            shutil.copy2(path, destination / "reviews" / path.name)
        for path in (ROOT / "reviews").glob("*.md"):
            shutil.copy2(path, destination / "reviews" / path.name)
        shutil.copy2(ROOT / "sources/official-lecture-order.json", destination / "sources/official-lecture-order.json")
        result = subprocess.run([sys.executable, "tools/build_book.py"], cwd=destination,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        if result.returncode:
            issues.append({"error": "source_free_build_failed", "stderr": result.stderr})
        compared = []
        for path in sorted(ROOT.glob("*.html")):
            other = destination / path.name
            match = other.exists() and digest(path) == digest(other)
            compared.append({"file": path.name, "bytes_match": match})
            if not match:
                issues.append({"error": "rebuilt_html_differs", "file": path.name})
        execution = subprocess.run(
            [sys.executable, "tools/check_book.py"], cwd=destination,
            capture_output=True, text=True, encoding="utf-8", timeout=1800,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        execution_report = json.loads(
            (destination / "validation/book-checks.json").read_text(encoding="utf-8"))
        if execution.returncode or not execution_report["passed"]:
            issues.append({"error": "source_free_execution_failed",
                           "stderr": execution.stderr,
                           "details": execution_report["issues"]})
        archive = dist / "predictive-modelling-notes.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(destination.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    name = path.relative_to(destination).as_posix()
                    if "private" in path.relative_to(destination).parts:
                        raise RuntimeError("A private path entered the export")
                    bundle.write(path, name)
        report = {"passed": not issues, "issues": issues,
                  "private_sources_present": False, "html_files_compared": len(compared),
                  "source_free_labs_executed": execution_report["labs_executed"],
                  "source_free_checks_passed": execution_report["passed"],
                  "files": compared, "archive": archive.relative_to(ROOT).as_posix(),
                  "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
                  "scope": "Reader rebuild and all main labs from authored public inputs without private media, transcripts or R references; no network retrieval in the build/lab code."}
    (ROOT / "validation/source-free-checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
