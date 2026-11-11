# Qwen2.5-7B Anatomy

An interactive, single-page look inside **Qwen2.5 7B Instruct**: how its 7.6 billion weights are arranged, and how it turns text into the next word. Every shape, storage format and value on the page was read directly from the model's GGUF file, the one Ollama downloads for `qwen2.5:7b`.

It is meant for people who are curious about how a language model works but have never looked inside one.

## What's on the page

| Tab | What it shows |
|---|---|
| **How it works (3D)** | A 9-step guided tour of one token, "is" in "The capital of France is", travelling from the embedding table up through all 28 layers to the model's real next-token probabilities. You can turn, move and zoom the scene, colour the layers by real per-layer measurements, and hover any layer or attention head to see its values. Also includes a glossary of 22 terms. |
| **Layer anatomy** | A diagram of one layer's 14 tensors with that layer's real numbers on every box, plus a whole-model view. Click a tensor for its shape, size, storage format, statistics and a 40 × 40 heatmap of its actual weights. |
| **Where the 7.6B live** | How the parameters split between feed-forward, attention, embeddings, norms and biases, and how many bytes each storage format uses. |
| **Attention heads** | A 28 × 28 grid of every query head (and the 4 key heads) across all layers, coloured by bias size. Shows grouped-query attention: 7 query heads share each key/value head. |
| **Layer by layer** | Charts comparing the 28 layers on norm gains, bias sizes and more, plus which layers are stored in the more precise Q6_K format. |
| **Token embeddings** | Real embedding rows for a handful of tokens, with a rough similarity comparison. |

## Open it

Open [`docs/index.html`](docs/index.html) in a browser. The page is self-contained apart from two things it loads from the internet: Google Fonts and [three.js](https://threejs.org/) r128 from cdnjs for the 3D view. Without a connection, the fonts fall back to system fonts and the 3D view shows a message instead, and everything else still works.

To publish it with **GitHub Pages**, go to the repository's Settings → Pages, choose **Deploy from a branch**, and select your main branch with the `/docs` folder.

## Regenerate the data yourself

The extracted data is committed in [`data/`](data/), so you don't need the model to view or change the page. To re-derive everything from the model file:

**Requirements:** Python 3.8 or newer (standard library only) and [Ollama](https://ollama.com) with the model pulled (about 4.7 GB):

```bash
ollama pull qwen2.5:7b
```

**1. Read the weights** (a few seconds). This finds Ollama's copy of the model automatically, or you can pass `--model path/to/file.gguf`:

```bash
python3 scripts/extract_weights.py
```

**2. Capture real next-token probabilities** (needs Ollama running, and an Ollama version that supports `logprobs`):

```bash
python3 scripts/capture_probs.py
```

**3. Build the page:**

```bash
python3 scripts/build.py
```

This writes `docs/index.html`. To change the page itself, edit [`src/page.html`](src/page.html) and run the build again.

## How it works

GGUF is the file format used by llama.cpp and Ollama. [`scripts/gguf_reader.py`](scripts/gguf_reader.py) is a small pure-Python reader for it, which:

- parses the header, the metadata (architecture, sizes, the 152,064-token vocabulary) and the table of 339 tensors;
- reads 32-bit float tensors (norm gains and biases) directly;
- unpacks ("dequantizes") the compressed **Q4_K** and **Q6_K** blocks back into ordinary numbers, using the same formulas as llama.cpp. Q4_K stores 256 weights in 144 bytes and Q6_K stores 256 weights in 210 bytes.

`extract_weights.py` then computes what the page shows: per-layer statistics over every norm and bias value, per-head maximum biases, weight statistics for every matrix, and the corner patches of each matrix used for the heatmaps.

## What's real and what's illustrative

- **Real, from the model file:** every tensor's name, shape and storage format, the norm and bias values, the per-head bias colours, the weight heatmaps and statistics, the token ids and the embedding rows.
- **Real, from running the model:** the next-token probabilities in the 3D tour, captured through Ollama. They vary by about a percentage point between runs and machines.
- **Sampled:** the weight statistics for each matrix (spread, mean and largest value) come from 8,192 values spread across that matrix, not from every weight. Statistics for norms and biases use every value.
- **Illustrative:** the thickness of the attention arcs and the moving dot in the 3D tour. A static page can't run a 7B model, and Ollama doesn't expose what happens inside the layers. The tour labels these parts on every step.

## Limitations

- The reader supports only the tensor types in this file: F32, Q4_K and Q6_K. Other quantizations of the model (for example Q8_0 or F16 downloads) would need more dequantization code.
- The scripts check that the file is a 28-layer `qwen2` model. The page's text and layout are written specifically for Qwen2.5-7B.

## Project layout

```
docs/index.html            the built page (served by GitHub Pages)
src/page.html              page source: HTML, CSS and JavaScript, with data placeholders
data/weights.json          everything read from the GGUF file
data/probabilities.json    next-token probabilities captured through Ollama
scripts/gguf_reader.py     GGUF parser and Q4_K / Q6_K dequantization
scripts/extract_weights.py builds data/weights.json from the model file
scripts/capture_probs.py   builds data/probabilities.json from a running Ollama
scripts/build.py           combines src/ and data/ into docs/index.html
```

## License and credits

The code in this repository (the page source, scripts and build) is released under the [MIT License](LICENSE).

[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) is developed by the Qwen team at Alibaba Cloud and released under the Apache License 2.0. The files in `data/` and the values embedded in `docs/index.html` are derived from its weights and remain under that license. See [NOTICE](NOTICE).

This project is not affiliated with or endorsed by Alibaba Cloud or the Qwen team.
