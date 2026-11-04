"""Exercise rendered local-file reader pages in desktop and mobile Chromium."""
import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", nargs="+", type=int)
    args = parser.parse_args()
    failures, checks = [], []
    paths = sorted(ROOT.glob("*.html"))
    if args.chapters:
        paths = [p for p in paths if p.stem.startswith("chapter-")
                 and int(p.stem.split("-")[1]) in args.chapters]
    chrome = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe"
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=str(chrome) if chrome.exists() else None, headless=True)
        for width, height in ((1440, 1000), (390, 844)):
            context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
            # Read the package offline: all http(s) access is intentionally blocked.
            context.route("http://**/*", lambda route: route.abort())
            context.route("https://**/*", lambda route: route.abort())
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            for path in paths:
                errors.clear()
                page.goto(path.as_uri(), wait_until="load")
                page.wait_for_function("window.MathJax && MathJax.startup && MathJax.startup.promise")
                page.evaluate("() => MathJax.startup.promise")
                page.wait_for_timeout(150)
                result = page.evaluate("""() => ({
                    mathExpressions: document.querySelectorAll('mjx-container').length,
                    mathErrors: [...document.querySelectorAll('mjx-merror, [data-mjx-error]')].map(e => e.textContent),
                    unknownCommands: [...document.querySelectorAll('mjx-container')].filter(e => /\\\\[A-Za-z]+/.test(e.textContent)).map(e => e.textContent),
                    overflow: document.documentElement.scrollWidth > innerWidth + 2,
                    h1: document.querySelectorAll('main h1').length,
                    solutions: [...document.querySelectorAll('main details summary')].filter(s => /solution|answer/i.test(s.textContent)).length
                })""")
                result.update(file=path.name, width=width, javascript_errors=list(errors))
                solution = page.locator("main details").first
                if solution.count():
                    summary = solution.locator("summary").first
                    summary.click()
                    result["disclosure_opens"] = solution.evaluate("el => el.open")
                    summary.click()
                    result["disclosure_closes"] = not solution.evaluate("el => el.open")
                before = page.evaluate("document.documentElement.dataset.theme")
                page.locator("[data-action=theme]").click()
                result["theme_changes"] = page.evaluate("document.documentElement.dataset.theme") != before
                button = page.locator("[data-action=larger]")
                size_before = page.evaluate("getComputedStyle(document.body).fontSize")
                button.click()
                result["font_size_changes"] = page.evaluate("getComputedStyle(document.body).fontSize") != size_before
                page.locator("[data-action=smaller]").click()
                page.locator("[data-action=theme]").click()
                if result["mathErrors"] or result["unknownCommands"] or result["overflow"] or result["h1"] != 1 or errors or not result["theme_changes"] or not result["font_size_changes"]:
                    failures.append(result)
                checks.append(result)
            page.goto((ROOT / "index.html").as_uri(), wait_until="load")
            page.locator("#course-search").fill("correlation")
            found = page.locator("#search-results li").count()
            if not found:
                failures.append({"file": "index.html", "width": width, "error": "search_no_results"})
            checks.append({"file": "index.html", "width": width, "search_results": found})
            if width == 1440:
                page.evaluate("window.scrollTo(0, 0)")
                page.screenshot(path=str(ROOT / "assets/reader-preview.png"), full_page=False)
            context.close()
        browser.close()
    report = {"checks": checks, "failures": failures, "passed": not failures,
              "network": "http_and_https_blocked; local file assets only",
              "scope": "All reader pages unless chapter selection is specified, at desktop/mobile widths; actual MathJax completion awaited."}
    (ROOT / "validation/reader-checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "failures": failures, "passed": not failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
