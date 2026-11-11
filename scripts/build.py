"""Combine src/page.html with the JSON in data/ into one self-contained page.

Usage:
    python3 scripts/build.py                     # writes docs/index.html
    python3 scripts/build.py --fragment OUT.html # page content without <html>/<head>/<body>
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = '\n<header class="top wrap">'  # everything before this goes into <head>

DOC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="An interactive look inside Qwen2.5 7B Instruct: a 3D tour of one token through the model and maps of its real weights.">
<style>:root{{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0;font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
{head}
</head>
<body>
{body}
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=ROOT / "docs" / "index.html", type=Path)
    ap.add_argument("--fragment", type=Path, help="also write the page without the document wrapper")
    args = ap.parse_args()

    page = (ROOT / "src" / "page.html").read_text()
    weights = json.loads((ROOT / "data" / "weights.json").read_text())
    probs = json.loads((ROOT / "data" / "probabilities.json").read_text())
    tour = {"prompt": probs["prompt"], "prompt_tokens": probs["prompt_tokens"], "tokens": weights.pop("tour_tokens"),
            "steps": probs["steps"], "ms_per_token": probs["ms_per_token"]}

    for marker in ("/*__DATA__*/null", "/*__LEARN__*/null"):
        if page.count(marker) != 1:
            raise SystemExit(f"src/page.html must contain {marker} exactly once")
    page = (page.replace("/*__DATA__*/null", json.dumps(weights, separators=(",", ":")))
                .replace("/*__LEARN__*/null", json.dumps(tour, separators=(",", ":"))))

    if args.fragment:
        args.fragment.write_text(page)
        print(f"wrote {args.fragment}")
    head, body = page.split(SPLIT, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(DOC.format(head=head.strip(), body=SPLIT.lstrip("\n") + body))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
