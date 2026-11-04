"""Build the portable reader from authored HTML; never generate chapter prose."""
import hashlib
import html
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(fragment):
    soup = BeautifulSoup(fragment, "html.parser")
    for tag in soup(["script", "style", "pre"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)


def merge_sources(manifest, metadata):
    readings = {r["video_id"]: r for meta in metadata.values()
                for r in meta.get("reading_records", [])}
    for entry in manifest["entries"]:
        video_id = entry["video_id"]
        acquisition = read_json(ROOT / "sources/private/acquisition" / f"{video_id}.json", {})
        if acquisition:
            entry["transcript_status"] = acquisition.get("status", "unknown")
            for key in ("transcript_sha256", "caption_sha256", "segments", "source_type"):
                if key in acquisition:
                    entry[key] = acquisition[key]
        reading = readings.get(video_id)
        if reading:
            entry["reading_status"] = "read_full" if reading.get("read_full") else "partial"
            entry["reading_record"] = reading
    write_json(ROOT / "source-manifest.json", manifest)


def review_status(number, body_hash):
    rounds = []
    for round_number in (1, 2):
        report = read_json(ROOT / "reviews" / f"ch{number:02d}-r{round_number}.json", {})
        report_hash = report.get("body_sha256") or report.get("chapter_sha256") or report.get("sha256")
        verdict = str(report.get("verdict", "pending")).lower()
        accepted = verdict in ("accept", "accepted") and report_hash == body_hash
        rounds.append({"round": round_number, "verdict": verdict,
                       "current_digest": report_hash == body_hash, "accepted": accepted})
    return {"rounds": rounds, "accepted": all(r["accepted"] for r in rounds)}


def shell(title, body, chapters, current=None, toc=""):
    navigation = "".join(
        '<li><a ' + ('aria-current="page" ' if c["number"] == current else "")
        + f'href="chapter-{c["number"]:02d}.html">{c["number"]}. {html.escape(c["title"])}</a></li>'
        for c in chapters)
    return f'''<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Independent study notes for NPTEL Predictive Modelling with Applications, with Python laboratories and worked solutions.">
<title>{html.escape(title)} — Predictive Modelling</title>
<link rel="stylesheet" href="assets/book.css">
<script defer src="assets/math-config.js"></script>
<script defer src="assets/mathjax-tex-svg.js"></script>
<script defer src="assets/search-index.js"></script>
<script defer src="assets/book.js"></script></head><body>
<a class="skip" href="#main">Skip to content</a>
<header class="toolbar"><a class="brand" href="index.html">PREDICTIVE MODELLING</a>
<div class="controls"><button type="button" data-action="theme">Theme</button>
<button type="button" data-action="smaller" aria-label="Smaller text">A−</button>
<button type="button" data-action="larger" aria-label="Larger text">A+</button>
<button type="button" data-action="print">Print / PDF</button></div></header>
<div class="layout"><aside class="sidebar"><details open><summary>Course contents</summary>
<ol>{navigation}</ol><nav class="utility" aria-label="Companion pages">
<a href="index.html">Course home and search</a><a href="study-guide.html">Study guide</a>
<a href="source-map.html">Recordings and timestamps</a><a href="python-conversions.html">R-to-Python conversions</a>
<a href="laboratory.html">Python laboratory</a><a href="notation.html">Notation</a>
<a href="about.html">Sources and verification</a></nav></details></aside>
<main id="main"><p class="eyebrow">IIT Hyderabad · NPTEL · Independent study edition</p>
<h1>{html.escape(title)}</h1>{toc}{body}
<footer>Based on Prof. Sayantee Jana’s NPTEL course. Independent educational notes; no institutional endorsement.
Source recordings remain with their creators. Mathematics and teaching additions are identified in context.</footer>
</main></div></body></html>'''


def laboratory(number):
    source = ROOT / "labs" / f"ch{number:02d}.py"
    stdout = ROOT / "labs" / f"ch{number:02d}-output.txt"
    if not source.exists():
        return ""
    output = stdout.read_text(encoding="utf-8") if stdout.exists() else "Execution output not yet recorded."
    helpers = sorted((ROOT / "labs").glob(f"ch{number:02d}_*.py"))
    supporting = ('<p>Supporting modules: ' + ' · '.join(
        f'<a href="labs/{html.escape(p.name)}"><code>{html.escape(p.name)}</code></a>'
        for p in helpers) + '.</p>') if helpers else ''
    return f'''<section id="python-laboratory"><h2>Python laboratory</h2>
<p>Run <a href="labs/ch{number:02d}.py"><code>labs/ch{number:02d}.py</code></a> from the course folder.
Data and figure paths, when used, are resolved by the program. Dependencies are listed in
<a href="requirements.txt">requirements.txt</a>.</p>{supporting}
<details><summary>Complete runnable Python source</summary><pre><code class="language-python">{html.escape(source.read_text(encoding="utf-8"))}</code></pre></details>
<details id="laboratory-output"><summary>Recorded execution output</summary><pre><code>{html.escape(output)}</code></pre></details></section>'''


def main():
    plan = read_json(ROOT / "plan/chapters.json")
    manifest = read_json(ROOT / "source-manifest.json")
    metadata = {c["number"]: read_json(ROOT / "validation/chapters" / f'ch{c["number"]:02d}.json', {})
                for c in plan["chapters"]}
    merge_sources(manifest, metadata)
    available = [c for c in plan["chapters"]
                 if metadata[c["number"]]
                 and (ROOT / "chapters" / f'ch{c["number"]:02d}.html').exists()
                 and (ROOT / "labs" / f'ch{c["number"]:02d}.py').exists()
                 and (ROOT / "labs" / f'ch{c["number"]:02d}-output.txt').exists()]
    records, search, examples = [], [], []
    for chapter in available:
        number = chapter["number"]
        path = ROOT / "chapters" / f"ch{number:02d}.html"
        fragment = path.read_text(encoding="utf-8")
        # Bodies render at the book root; allow authoring links from chapters/.
        fragment = re.sub(r'(["\x27])\.\./(labs|assets)/', r'\1\2/', fragment)
        soup = BeautifulSoup(fragment, "html.parser")
        heads = [h for h in soup.find_all(["h2", "h3"]) if h.get("id")]
        toc = '<nav class="toc" aria-label="Chapter sections"><ul>' + "".join(
            f'<li><a href="#{html.escape(h["id"], quote=True)}">{html.escape(h.get_text(" ", strip=True))}</a></li>' for h in heads) + "</ul></nav>"
        source_links = " · ".join(f'<a href="https://www.youtube.com/watch?v={v}">Recording {i+1}</a>'
                                  for i, v in enumerate(chapter["videos"]))
        status = review_status(number, digest(path))
        review_label = "Two current-content independent reviews accepted" if status["accepted"] else "Editorial review in progress"
        meta = f'<p class="meta">Chapter {number} · {source_links}</p><p class="meta">{review_label}</p>'
        before = f'<a href="chapter-{number-1:02d}.html">← Previous chapter</a>' if any(c["number"] == number-1 for c in available) else '<a href="index.html">← Course home</a>'
        after = f'<a href="chapter-{number+1:02d}.html">Next chapter →</a>' if any(c["number"] == number+1 for c in available) else '<a href="index.html">Course home →</a>'
        body = meta + fragment + laboratory(number) + f'<nav class="chapter-nav" aria-label="Chapter navigation">{before}{after}</nav>'
        (ROOT / f"chapter-{number:02d}.html").write_text(shell(chapter["title"], body, available, number, toc), encoding="utf-8")
        text = plain(fragment)
        solved = sum("solution" in s.get_text().lower() or "answer" in s.get_text().lower() for s in soup.find_all("summary"))
        record = {"number": number, "title": chapter["title"], "body_sha256": digest(path),
                  "words_approx": len(text.split()), "solved_exercises": solved,
                  "source_videos": chapter["videos"], "review": status,
                  "lab_present": (ROOT / "labs" / f"ch{number:02d}.py").exists()}
        records.append(record)
        search.append({"number": number, "title": chapter["title"], "url": f"chapter-{number:02d}.html", "text": text})
        for i, example in enumerate(metadata[number].get("r_examples", []), 1):
            examples.append({"chapter": number, "example_index": i, **example})
    (ROOT / "assets/search-index.js").write_text("window.COURSE_SEARCH = " + json.dumps(search, ensure_ascii=False).replace("</", "<\\/") + ";\n", encoding="utf-8")
    complete_draft = len(records) == len(plan["chapters"])
    reviewed = complete_draft and all(r["review"]["accepted"] for r in records)
    phase = ("complete_reviewed_edition" if reviewed else
             "complete_draft_under_review" if complete_draft else "in_progress")
    completion = {"phase": phase, "planned_chapters": len(plan["chapters"]),
                  "written_chapters": len(records), "accepted_chapters": sum(r["review"]["accepted"] for r in records),
                  "teaching_recordings": 39, "playlist_entries": 40,
                  "written_recording_coverage": len(set(v for r in records for v in r["source_videos"])),
                  "words_approx": sum(r["words_approx"] for r in records),
                  "solved_exercises": sum(r["solved_exercises"] for r in records),
                  "labs_present": sum(r["lab_present"] for r in records),
                  "r_examples_inventoried": len(examples), "chapters": records,
                  "publication_performed": bool(read_json(
                      ROOT / "validation/publication-checks.json", {}).get("pages_verified")),
                  "limitations": ["Machine-caption source text requires correction; selected frames are not a comprehensive audiovisual audit.",
                                  "Independent AI reviews are not human certification or formal proof verification."]}
    write_json(ROOT / "completion.json", completion)
    write_json(ROOT / "validation/r-to-python-examples.json", {
        "inventory_status": "assembled_from_chapter_reading_records",
        "coverage": {"teaching_recordings": 39,
                     "read_full": sum(e["reading_status"] == "read_full" for e in manifest["entries"] if e["role"] == "teaching")},
        "all_inventoried_methods_implemented": bool(examples) and all(
            str(e.get("status", "")).startswith(("verified_", "method_converted_",
                                                   "converted_", "implemented_and_verified_"))
            for e in examples),
        "all_original_numerical_outputs_reproduced": False,
        "examples": examples,
        "limitations": ["Method conversion and exact source-data numerical reproduction have separate statuses; inspect individual records."]})
    pages = {}
    intro = (ROOT / "frontmatter/home.html").read_text(encoding="utf-8")
    metrics = f'<div class="metrics"><span>{len(records)}/{len(plan["chapters"])} chapters</span><span>{completion["written_recording_coverage"]}/39 teaching lectures</span><span>{completion["words_approx"]:,} words, approximately</span><span>{completion["solved_exercises"]} worked solutions</span><span>{completion["labs_present"]} Python labs</span></div>'
    cards = '<ol class="chapters">' + "".join(f'<li><a class="chapter-title" href="chapter-{c["number"]:02d}.html">{c["number"]}. {html.escape(c["title"])}</a><p class="small">{len(c["videos"])} source recording(s) · Python laboratory and worked solutions</p></li>' for c in available) + '</ol>'
    pages["index.html"] = ("Predictive Modelling with Applications", intro + metrics + '<h2 id="chapters">The course</h2>' + cards + '<h2 id="search">Search the notes</h2><label for="course-search">Search all authored chapters</label><input type="search" id="course-search" placeholder="For example: leverage, varimax, odds"><div id="search-results" class="search-results" aria-live="polite"></div>')
    for name, title in (("study-guide", "Study guide"), ("about", "Sources and verification")):
        pages[f"{name}.html"] = (title, (ROOT / "frontmatter" / f"{name}.html").read_text(encoding="utf-8"))
    rows = []
    for entry in sorted(manifest["entries"], key=lambda e: e["teaching_position"] or 0):
        number = entry["chapter"]
        chapter_link = f'<a href="chapter-{number:02d}.html">{number}</a>' if number and any(c["number"] == number for c in available) else "Orientation" if not number else "Pending"
        duration = f'{entry["duration_seconds"]//60}:{entry["duration_seconds"]%60:02d}'
        rows.append(f'<tr><td>{entry["teaching_position"] or "—"}</td><td>{entry["playlist_position"]}</td><td><a href="{entry["url"]}">{html.escape(entry["title"])}</a></td><td>{duration}</td><td>{chapter_link}</td><td>{html.escape(entry["reading_status"])}</td></tr>')
    pages["source-map.html"] = ("Recordings and source coverage", '<p>The playlist is shuffled. Teaching order was verified against the official NPTEL course page’s lesson array. Original upload positions remain visible. Section anchors inside chapters identify the related source discussion; authored derivations and corrections remain our additions.</p><p><a href="source-manifest.json">Machine-readable source manifest</a> · <a href="sources/official-lecture-order.json">Official-order evidence</a></p><table><thead><tr><th>Lecture</th><th>Playlist position</th><th>Original recording title</th><th>Duration</th><th>Chapter</th><th>Text reading</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>')
    conversion_rows = []
    for example in examples:
        method = example.get("source_method", example.get("r_calls", example.get("source_description", example.get("id", "Example"))))
        converted = example.get("python_method", example.get("python_equivalent", "See chapter laboratory"))
        evidence = example.get("verification", "See chapter record")
        limits = str(example.get("data_status", "")) + ' ' + str(example.get("limitations", ""))
        conversion_rows.append(f'<tr><td><a href="chapter-{example["chapter"]:02d}.html">{example["chapter"]}</a></td><td>{html.escape(str(method))}</td><td>{html.escape(str(converted))}</td><td>{html.escape(str(example.get("status", "unverified")))}</td><td>{html.escape(str(evidence))}<br>{html.escape(str(limits))}</td></tr>')
    conversions_intro = (ROOT / "frontmatter/python-conversions.html").read_text(encoding="utf-8")
    pages["python-conversions.html"] = ("R examples converted to Python", conversions_intro + '<p><a href="validation/r-to-python-examples.json">Complete conversion ledger</a></p><table><thead><tr><th>Chapter</th><th>Source R example</th><th>Python implementation</th><th>Status</th><th>Evidence and limits</th></tr></thead><tbody>' + "".join(conversion_rows) + '</tbody></table>')
    lab_rows = "".join(f'<tr><td><a href="chapter-{c["number"]:02d}.html#python-laboratory">{c["number"]}. {html.escape(c["title"])}</a></td><td><a href="labs/ch{c["number"]:02d}.py">Python source</a></td><td><a href="labs/ch{c["number"]:02d}-output.txt">Recorded output</a></td></tr>' for c in available if (ROOT / "labs" / f'ch{c["number"]:02d}.py').exists())
    pages["laboratory.html"] = ("Python laboratory", (ROOT / "frontmatter/laboratory.html").read_text(encoding="utf-8") + '<table><thead><tr><th>Chapter</th><th>Program</th><th>Execution evidence</th></tr></thead><tbody>' + lab_rows + '</tbody></table>')
    notation_text = (ROOT / "plan/notation.md").read_text(encoding="utf-8")
    table_lines = [line for line in notation_text.splitlines() if line.startswith("|")]
    notation_rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(v.strip())}</td>' for v in line.strip('|').split('|')) + '</tr>' for line in table_lines if not re.match(r'^\|[- |]+\|$', line))
    pages["notation.html"] = ("Notation reference", '<p>Local definitions in each chapter govern every calculation. This table records the shared contract; chapter-specific notation is introduced where used.</p><table>' + notation_rows + '</table>')
    plan_body = '<p>Autonomous production covers all 39 teaching recordings, grouped by topic. All R examples are to be converted to Python, with specific explanations for obstacles.</p>' + ''.join(f'<section id="plan-{c["number"]}"><h2>{c["number"]}. {html.escape(c["title"])}</h2><ul>' + ''.join(f'<li>{html.escape(r)}</li>' for r in c["results"]) + f'</ul><p>{html.escape(c["lab"])}</p></section>' for c in plan["chapters"])
    pages["course-plan.html"] = ("Course architecture", plan_body)
    for name, (title, body) in pages.items():
        (ROOT / name).write_text(shell(title, body, available), encoding="utf-8")
    print(json.dumps({k: v for k, v in completion.items() if k != "chapters"}, indent=2))


if __name__ == "__main__":
    main()
