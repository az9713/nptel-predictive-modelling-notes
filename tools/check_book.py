"""Check the actual reader, source bounds, exact embeds, and executable labs."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def seconds(value):
    if str(value).isdigit():
        return int(value)
    matches = re.findall(r"(\d+)([hms])", str(value))
    return sum(int(n) * {"h": 3600, "m": 60, "s": 1}[unit] for n, unit in matches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-labs", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source-manifest.json").read_text(encoding="utf-8"))
    bounds = {e["video_id"]: e["duration_seconds"] for e in manifest["entries"]}
    issues, links, timestamps, pages, executions = [], 0, 0, [], []
    for path in sorted(ROOT.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        if "\ufffd" in text:
            issues.append({"file": path.name, "error": "unicode_replacement_character"})
        soup = BeautifulSoup(text, "html.parser")
        ids = [tag["id"] for tag in soup.find_all(id=True)]
        if len(ids) != len(set(ids)):
            issues.append({"file": path.name, "error": "duplicate_ids"})
        if not soup.html or soup.html.get("lang") != "en":
            issues.append({"file": path.name, "error": "missing_english_document_language"})
        for tag in soup.find_all(["a", "script", "link", "img"]):
            raw = tag.get("href") if tag.name in ("a", "link") else tag.get("src")
            if not raw:
                continue
            url = urlsplit(raw)
            if url.scheme in ("javascript", "data"):
                issues.append({"file": path.name, "error": "unexpected_active_url", "url": raw})
            if url.scheme in ("http", "https"):
                if "youtube.com" in url.netloc:
                    query = parse_qs(url.query)
                    video = query.get("v", [""])[0]
                    if video in bounds and ("t" in query or "start" in query):
                        start = seconds(query.get("t", query.get("start"))[0])
                        timestamps += 1
                        if start < 0 or start > bounds[video]:
                            issues.append({"file": path.name, "error": "timestamp_out_of_bounds", "url": raw})
                continue
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            links += 1
            if not target.is_relative_to(ROOT):
                issues.append({"file": path.name, "error": "link_outside_package", "url": raw})
            elif not target.exists():
                issues.append({"file": path.name, "error": "missing_link_target", "url": raw})
            elif url.fragment and target.suffix == ".html":
                content = soup if target == path else BeautifulSoup(target.read_text(encoding="utf-8"), "html.parser")
                if not content.find(id=unquote(url.fragment)):
                    issues.append({"file": path.name, "error": "missing_fragment", "url": raw})
        number_match = re.fullmatch(r"chapter-(\d{2})\.html", path.name)
        if number_match:
            number = int(number_match[1])
            lab = ROOT / "labs" / f"ch{number:02d}.py"
            embedded = soup.select_one("#python-laboratory code.language-python")
            if not lab.exists() or not embedded or embedded.get_text() != lab.read_text(encoding="utf-8"):
                issues.append({"file": path.name, "error": "lab_embed_mismatch"})
            output = ROOT / "labs" / f"ch{number:02d}-output.txt"
            embedded_output = soup.select_one("#laboratory-output pre code")
            if not output.exists() or not embedded_output or embedded_output.get_text() != output.read_text(encoding="utf-8"):
                issues.append({"file": path.name, "error": "lab_output_embed_mismatch"})
        pages.append({"file": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not args.skip_labs:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for lab in sorted((ROOT / "labs").glob("ch[0-9][0-9].py")):
            record = {"path": lab.relative_to(ROOT).as_posix()}
            try:
                run = subprocess.run([sys.executable, str(lab)], cwd=ROOT, env=env,
                                     capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", timeout=240)
                expected = lab.with_name(lab.stem + "-output.txt")
                match = expected.exists() and run.stdout.strip() == expected.read_text(encoding="utf-8").strip()
                record.update(exit_code=run.returncode, stdout_matches_saved=match,
                              stdout=run.stdout, stderr=run.stderr)
                if run.returncode or not match:
                    issues.append({"file": record["path"], "error": "lab_failed_or_output_changed"})
            except subprocess.TimeoutExpired:
                record["error"] = "timeout_240_seconds"
                issues.append({"file": record["path"], "error": "lab_timeout"})
            executions.append(record)
    report = {"html_pages": len(pages), "local_links_checked": links,
              "source_timestamps_checked": timestamps, "pages": pages,
              "labs_executed": len(executions), "executions": executions,
              "issues": issues, "passed": not issues,
              "limits": "Mechanical, source-bound, and finite execution checks; not scientific acceptance."}
    target = ROOT / "validation" / ("static-checks.json" if args.skip_labs else "book-checks.json")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("html_pages", "local_links_checked", "source_timestamps_checked", "labs_executed", "issues", "passed")}, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
