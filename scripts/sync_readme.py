"""Copy generated tables into README.md so that its numbers cannot drift from the results.

    python scripts/sync_readme.py           # rewrite the marked blocks
    python scripts/sync_readme.py --check   # exit 1 if a block differs from its source

A block in the README looks like

    <!-- table: summary.md#Main result -->
    ...copied section body...
    <!-- /table -->

and is replaced by the body of the section "## Main result" of results/baseline/summary.md.
"""

from __future__ import annotations

import argparse
import re
import sys

from emoswitch.config import ROOT, load_config

BLOCK = re.compile(r"(<!-- table: (?P<file>[\w.]+)#(?P<title>[^>]+?) -->\n)(?P<body>.*?)(<!-- /table -->)", re.S)


def section(results_dir, file: str, title: str) -> str:
    text = (results_dir / file).read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.M)
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        if head.strip() == title:
            # links in the generated files are relative to results/<name>/; the README sits at the root
            return body.strip().replace("](../../", "](") + "\n"
    raise SystemExit(f"[sync] no section {title!r} in {file}")


def render(readme: str, results_dir) -> str:
    return BLOCK.sub(lambda m: m.group(1) + section(results_dir, m["file"], m["title"]) + m.group(5), readme)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    results_dir = ROOT / "results" / load_config("configs/baseline.yaml")["name"]
    path = ROOT / "README.md"
    readme = path.read_text(encoding="utf-8")
    synced = render(readme, results_dir)
    if args.check:
        sys.exit(0 if synced == readme else "[sync] README.md is out of date: run python scripts/sync_readme.py")
    path.write_text(synced, encoding="utf-8")
    print(f"[sync] {len(BLOCK.findall(readme))} blocks written")


if __name__ == "__main__":
    main()
