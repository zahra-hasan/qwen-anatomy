"""Read Qwen2.5-7B's GGUF file and write every number the page shows to data/weights.json.

Usage:
    python3 scripts/extract_weights.py                 # finds the qwen2.5:7b file Ollama downloaded
    python3 scripts/extract_weights.py --model PATH    # or point at any copy of the GGUF file

Takes a few seconds and needs no third-party packages.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUF, find_ollama_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Tokens shown in the "Token embeddings" tab (first 64 numbers of each row).
EMBED_SAMPLE = ["ĠParis", "ĠFrance", "ĠLondon", "Ġocean", "Ġsea", "Ġthe", "Ġcat", "Ġdog", "1", "7", "<|im_start|>"]
# The 3D tour's prompt, "The capital of France is", as Qwen2.5 tokenizes it, plus the
# token the model picks next. Full 3,584-number rows are stored for these.
TOUR_TOKENS = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis", "ĠParis"]
PATCH = 40  # weight heatmaps show the top-left PATCH x PATCH corner of each matrix
SAMPLES = 32  # weight statistics use SAMPLES blocks of 256 values spread across each matrix


def mean(v):
    return sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to the Qwen2.5-7B GGUF file (default: Ollama's qwen2.5:7b)")
    ap.add_argument("--out", default=ROOT / "data" / "weights.json", type=Path)
    args = ap.parse_args()

    path = Path(args.model) if args.model else find_ollama_model("qwen2.5", "7b")
    print(f"reading {path}")
    g = GGUF(path)
    if g.meta.get("general.architecture") != "qwen2" or g.meta.get("qwen2.block_count") != 28:
        sys.exit("This does not look like Qwen2.5-7B (expected architecture qwen2 with 28 layers).")
    n_layers = g.meta["qwen2.block_count"]

    tensors = [{"name": n, "dims": g.tensors[n]["dims"], "type": g.type_name(n), "bytes": g.n_bytes(n)} for n in g.order]

    # Per-layer statistics for the norm gains and attention biases (every value).
    layers = []
    for i in range(n_layers):
        an, fn = g.f32(f"blk.{i}.attn_norm.weight"), g.f32(f"blk.{i}.ffn_norm.weight")
        qb, kb, vb = (g.f32(f"blk.{i}.attn_{x}.bias") for x in "qkv")
        layers.append({
            "attn_norm_mean": mean(an), "attn_norm_max": max(an),
            "ffn_norm_mean": mean(fn), "ffn_norm_max": max(fn),
            "q_bias_maxabs": max(map(abs, qb)), "k_bias_maxabs": max(map(abs, kb)),
            "v_bias_maxabs": max(map(abs, vb)),
            "outliers": sum(abs(x) > 10 for x in qb + kb),
        })

    # Largest |bias| per attention head: 28 query heads and 4 key heads of 128 values.
    hd = 128
    qhead = [[round(max(abs(x) for x in g.f32(f"blk.{i}.attn_q.bias")[h * hd:(h + 1) * hd]), 2) for h in range(28)]
             for i in range(n_layers)]
    khead = [[round(max(abs(x) for x in g.f32(f"blk.{i}.attn_k.bias")[h * hd:(h + 1) * hd]), 2) for h in range(4)]
             for i in range(n_layers)]

    # Sampled statistics for every weight matrix.
    wstats = {}
    for name in g.order:
        dims = g.tensors[name]["dims"]
        if len(dims) != 2:
            continue
        n_blocks, n_rows = dims[0] // 256, dims[1]
        vals = []
        for j in range(SAMPLES):
            vals += g.block(name, (j * 7919) % n_rows, (j * 5) % n_blocks)
        m = mean(vals)
        wstats[name] = {"std": (sum((x - m) ** 2 for x in vals) / len(vals)) ** .5,
                        "maxabs": max(map(abs, vals)), "meanabs": mean([abs(x) for x in vals]),
                        "mean": m, "n": len(vals)}

    # Real weight patches (values x 1000, rounded) for every layer's seven matrices.
    patches = {}
    for i in range(n_layers):
        for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            name = f"blk.{i}.{t}.weight"
            patches[name] = [[round(v * 1000) for v in g.block(name, r, 0)[:PATCH]] for r in range(PATCH)]

    vocab = g.meta["tokenizer.ggml.tokens"]
    token_id = {t: i for i, t in enumerate(vocab)}
    emb = [{"token": w.replace("Ġ", "␣"), "id": token_id[w],
            "row": [round(v * 1000) for v in g.block("token_embd.weight", token_id[w], 0)[:64]]}
           for w in EMBED_SAMPLE if w in token_id]
    tour_tokens = [{"text": w.replace("Ġ", " "), "id": token_id[w],
                    "row": [round(v * 1000) for v in g.row("token_embd.weight", token_id[w])]}
                   for w in TOUR_TOKENS]

    meta = {k: v for k, v in g.meta.items() if not isinstance(v, list) and k != "tokenizer.chat_template"}
    out = {"meta": meta, "tensors": tensors, "layers": layers, "qhead": qhead, "khead": khead,
           "patches": patches, "wstats": wstats, "emb": emb, "vocab_sample": vocab[1000:1012],
           "tour_tokens": tour_tokens}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
