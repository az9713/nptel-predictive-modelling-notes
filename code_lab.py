"""List or execute a course laboratory selected by its chapter number."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("chapter", nargs="?", type=int, choices=range(1, 19))
args = parser.parse_args()
if args.chapter is None:
    for path in sorted((ROOT / "labs").glob("ch[0-9][0-9].py")):
        print(f"{int(path.stem[2:]):2d}  {path.name}")
else:
    path = ROOT / "labs" / f"ch{args.chapter:02d}.py"
    if not path.exists():
        parser.error("This laboratory has not yet been written.")
    raise SystemExit(subprocess.call([sys.executable, str(path)], cwd=ROOT))
