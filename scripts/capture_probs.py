"""Ask a running Ollama for Qwen2.5-7B's real next-token probabilities and save them to
data/probabilities.json. These fill steps 8 and 9 of the 3D tour.

Usage (Ollama must be running and `ollama pull qwen2.5:7b` done):
    python3 scripts/capture_probs.py
    python3 scripts/capture_probs.py --host http://localhost:11434 --model qwen2.5:7b

The prompt is fixed to match the tokens in scripts/extract_weights.py (TOUR_TOKENS).
Probabilities differ slightly between runs and machines.
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"


def post(host, path, body):
    req = urllib.request.Request(host.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--out", default=ROOT / "data" / "probabilities.json", type=Path)
    args = ap.parse_args()

    try:
        r = post(args.host, "/api/generate", {
            "model": args.model, "prompt": PROMPT, "raw": True, "stream": False,
            "logprobs": True, "top_logprobs": 8,
            "options": {"temperature": 0, "num_predict": 2, "seed": 0}})
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Ollama at {args.host} ({e}). Start Ollama and try again.")
    if not r.get("logprobs"):
        sys.exit("Ollama returned no logprobs. Update Ollama to a version that supports the logprobs option.")
    try:
        version = json.load(urllib.request.urlopen(args.host.rstrip("/") + "/api/version", timeout=10)).get("version")
    except Exception:
        version = None

    steps = [{"chosen": lp["token"], "top": [[t["token"], round(math.exp(t["logprob"]), 4)] for t in lp["top_logprobs"]]}
             for lp in r["logprobs"]]
    out = {"prompt": PROMPT, "prompt_tokens": r["prompt_eval_count"], "steps": steps,
           "ms_per_token": round(r["eval_duration"] / max(1, r["eval_count"]) / 1e6, 1),
           "captured_with": {"model": args.model, "ollama_version": version}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for i, s in enumerate(steps):
        print(f"token {i + 1}: " + ", ".join(f"{t!r} {p:.1%}" for t, p in s["top"][:4]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
