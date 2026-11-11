"""Minimal, dependency-free reader for GGUF model files (the format Ollama and llama.cpp use).

It understands everything in Qwen2.5-7B Q4_K_M: metadata, the tensor table, and the
F32, Q4_K and Q6_K tensor types. Quantized weights are unpacked ("dequantized") back into
ordinary floats with the same formulas llama.cpp uses.
"""
import json
import os
import struct
from pathlib import Path

# ggml type id -> (name, values per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 12: ("Q4_K", 256, 144), 14: ("Q6_K", 256, 210)}
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class GGUF:
    """Opens a GGUF file and exposes its metadata and tensors.

    meta     dict of metadata keys -> values (arrays become Python lists)
    tensors  dict of tensor name -> {"dims": [...], "type": id, "offset": int}
    order    tensor names in file order
    """

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        if self.f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        (self.version,) = self._rd("I")
        (n_tensors,) = self._rd("Q")
        (n_kv,) = self._rd("Q")
        self.meta = {}
        for _ in range(n_kv):
            key = self._str()
            (t,) = self._rd("I")
            self.meta[key] = self._val(t)
        self.tensors, self.order = {}, []
        for _ in range(n_tensors):
            name = self._str()
            (nd,) = self._rd("I")
            dims = list(self._rd("Q" * nd))
            (tt,) = self._rd("I")
            (off,) = self._rd("Q")
            self.tensors[name] = {"dims": dims, "type": tt, "offset": off}
            self.order.append(name)
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self.f.tell() + align - 1) // align * align

    # -- low-level parsing ---------------------------------------------------
    def _rd(self, fmt):
        return struct.unpack("<" + fmt, self.f.read(struct.calcsize(fmt)))

    def _str(self):
        (n,) = self._rd("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        if t in _SCALAR:
            return self._rd(_SCALAR[t])[0]
        if t == 8:
            return self._str()
        if t == 9:
            (et,) = self._rd("I")
            (n,) = self._rd("Q")
            return [self._val(et) for _ in range(n)]
        raise ValueError(f"unknown metadata type {t}")

    # -- tensor access -------------------------------------------------------
    def type_name(self, name):
        return TYPES[self.tensors[name]["type"]][0]

    def n_params(self, name):
        n = 1
        for d in self.tensors[name]["dims"]:
            n *= d
        return n

    def n_bytes(self, name):
        _, per_block, block_bytes = TYPES[self.tensors[name]["type"]]
        return self.n_params(name) // per_block * block_bytes

    def f32(self, name):
        """All values of a 1-D F32 tensor (norm gains, biases)."""
        t = self.tensors[name]
        assert t["type"] == 0, f"{name} is not F32"
        n = t["dims"][0]
        self.f.seek(self.data_start + t["offset"])
        return list(self._rd(f"{n}f"))

    def block(self, name, row, index):
        """The 256 dequantized values of block `index` in row `row` of a Q4_K/Q6_K matrix.

        GGUF dims are [inputs, outputs]: each row holds one output unit's inputs, stored
        as dims[0] / 256 consecutive blocks.
        """
        t = self.tensors[name]
        _, per_block, block_bytes = TYPES[t["type"]]
        row_bytes = t["dims"][0] // per_block * block_bytes
        self.f.seek(self.data_start + t["offset"] + row * row_bytes + index * block_bytes)
        return dequantize(t["type"], self.f.read(block_bytes))

    def row(self, name, row):
        """A whole dequantized row of a Q4_K/Q6_K matrix."""
        t = self.tensors[name]
        _, per_block, block_bytes = TYPES[t["type"]]
        row_bytes = t["dims"][0] // per_block * block_bytes
        self.f.seek(self.data_start + t["offset"] + row * row_bytes)
        return dequantize(t["type"], self.f.read(row_bytes))


# -- dequantization (formulas from llama.cpp's ggml-quants.c) -----------------
def dequantize(type_id, buf):
    if type_id == 12:
        return dequant_q4k(buf)
    if type_id == 14:
        return dequant_q6k(buf)
    raise ValueError(f"dequantizing type {type_id} is not supported")


def _scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequant_q4k(buf):
    """Q4_K: 256 weights in 144 bytes. A super-block scale and min (fp16), 8 sub-block
    scales and mins (6 bits each), then 4-bit values: w = d * scale * q - dmin * min."""
    out = []
    for b in range(len(buf) // 144):
        blk = buf[b * 144:(b + 1) * 144]
        d, dmin = struct.unpack("<ee", blk[:4])
        sc, qs = blk[4:16], blk[16:]
        is_ = 0
        for j in range(0, 256, 64):
            q = qs[j // 2: j // 2 + 32]
            s1, m1 = _scale_min_k4(is_, sc)
            s2, m2 = _scale_min_k4(is_ + 1, sc)
            out += [d * s1 * (x & 0xF) - dmin * m1 for x in q]
            out += [d * s2 * (x >> 4) - dmin * m2 for x in q]
            is_ += 2
    return out


def dequant_q6k(buf):
    """Q6_K: 256 weights in 210 bytes. 6-bit values split into low 4 bits (ql) and high
    2 bits (qh), 16 signed 8-bit sub-block scales and one fp16 scale: w = d * scale * (q - 32)."""
    out = []
    for b in range(len(buf) // 210):
        blk = buf[b * 210:(b + 1) * 210]
        ql, qh = blk[:128], blk[128:192]
        sc = struct.unpack("<16b", blk[192:208])
        (d,) = struct.unpack("<e", blk[208:210])
        y = [0.0] * 256
        for n in range(2):
            L, H, s = ql[64 * n:], qh[32 * n:], sc[8 * n:]
            for l in range(32):
                i = l // 16
                q1 = ((L[l] & 0xF) | ((H[l] & 3) << 4)) - 32
                q2 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
                q3 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
                q4 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
                base = 128 * n
                y[base + l] = d * s[i] * q1
                y[base + l + 32] = d * s[i + 2] * q2
                y[base + l + 64] = d * s[i + 4] * q3
                y[base + l + 96] = d * s[i + 6] * q4
        out += y
    return out


# -- locating the model ---------------------------------------------------------
def find_ollama_model(name="qwen2.5", tag="7b"):
    """Path of the weights file Ollama stored for `name:tag`.

    Ollama keeps a small manifest per model that points at content-addressed blobs.
    Honours the OLLAMA_MODELS environment variable, like Ollama itself.
    """
    root = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest.exists():
        raise FileNotFoundError(
            f"No Ollama manifest at {manifest}. Run `ollama pull {name}:{tag}` first, "
            f"or pass the GGUF file with --model.")
    layers = json.loads(manifest.read_text())["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"] == "application/vnd.ollama.image.model")
    return root / "blobs" / digest.replace(":", "-")
