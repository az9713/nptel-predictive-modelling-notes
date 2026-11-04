"""Acquire course captions and retain a timestamped, unabridged text record."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "sources" / "private"


def acquire(video_id):
    """Return an acquisition record; never infer that an acquired source was read."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid video identifier")
    capdir = PRIVATE / "captions"
    textdir = PRIVATE / "transcripts"
    logdir = PRIVATE / "acquisition"
    for directory in (capdir, textdir, logdir):
        directory.mkdir(parents=True, exist_ok=True)
    existing = list(capdir.glob(f"{video_id}.*.json3"))
    seed = PRIVATE / f"{video_id}.en-orig.json3"
    if not existing and seed.exists():
        destination = capdir / seed.name
        destination.write_bytes(seed.read_bytes())
        existing = [destination]
    result = None
    if not existing:
        command = [
            "yt-dlp", "--no-simulate", "--skip-download", "--write-subs",
            "--write-auto-subs", "--sub-langs", "en-orig,en", "--sub-format",
            "json3", "--no-playlist", "--no-warnings", "--socket-timeout",
            "20", "--retries", "1", "--sleep-subtitles", "1", "--paths",
            str(capdir), "-o", "%(id)s.%(ext)s",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=210)
            (logdir / f"{video_id}.log").write_text(
                result.stdout + "\n" + result.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            (logdir / f"{video_id}.log").write_text(str(exc), encoding="utf-8")
        existing = list(capdir.glob(f"{video_id}.*.json3"))
    record = {"video_id": video_id, "status": "unavailable", "read": False}
    if existing:
        chosen = sorted(existing, key=lambda p: ("en-orig" not in p.name, p.name))[0]
        try:
            data = json.loads(chosen.read_text(encoding="utf-8"))
            lines = []
            last_end = 0.0
            for event in data.get("events", []):
                content = "".join(s.get("utf8", "") for s in event.get("segs", []))
                content = " ".join(content.split())
                if not content:
                    continue
                start = event.get("tStartMs", 0) / 1000
                end = start + event.get("dDurationMs", 0) / 1000
                lines.append(f"[{start:.3f}s] {content}")
                last_end = max(last_end, end)
            if not lines:
                raise ValueError("Caption file contained no text")
            target = textdir / f"{video_id}.txt"
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            record.update(status="acquired", caption=str(chosen.relative_to(ROOT)),
                          transcript=str(target.relative_to(ROOT)),
                          caption_sha256=hashlib.sha256(chosen.read_bytes()).hexdigest(),
                          transcript_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          segments=len(lines), last_caption_end_seconds=last_end,
                          source_type="youtube_english_caption_track",
                          limitations="Caption/ASR text; equations, code, and visuals require source checks.")
        except (ValueError, KeyError) as exc:
            record["error"] = str(exc)
    (logdir / f"{video_id}.json").write_text(json.dumps(record, indent=2) + "\n",
                                            encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", nargs="+", type=int)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    plan = json.loads((ROOT / "plan" / "chapters.json").read_text(encoding="utf-8"))
    ids = args.ids or [v for c in plan["chapters"]
                      if not args.chapters or c["number"] in args.chapters
                      for v in c["videos"]]
    for video_id in ids:
        print(json.dumps(acquire(video_id)), flush=True)


if __name__ == "__main__":
    main()
