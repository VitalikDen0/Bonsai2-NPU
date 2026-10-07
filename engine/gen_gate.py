#!/usr/bin/env python3
"""Generate GATE1 reference vectors: float64 GEMV of 1-bit rows from safetensors.

Usage: python gen_gate.py --input model.safetensors --output gate.bin [--rows 64] [--seed 0]
"""
import argparse
import struct
import sys

import numpy as np
from safetensors import safe_open

MAGIC = b"GATE1"
VERSION = 0
GROUP = 128

# Representative coverage: embed, head, early/mid/late layers, attn + mlp + linear
WANT = [
    "language_model.model.embed_tokens",
    "language_model.lm_head",
]
for L in (0, 31, 63):
    p = f"language_model.model.layers.{L}."
    WANT += [p + "mlp.gate_proj", p + "mlp.down_proj", p + "mlp.up_proj",
             p + "linear_attn.in_proj_qkv", p + "linear_attn.out_proj"]
for L in (3, 35, 59):
    p = f"language_model.model.layers.{L}."
    WANT += [p + "self_attn.q_proj", p + "self_attn.o_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    f = safe_open(args.input, framework="np")
    keys = set(f.keys())
    avail = [b for b in WANT if b + ".weight" in keys and b + ".scales" in keys]
    if not avail:
        print("no wanted tensors found", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    # round-robin rows across tensors for even coverage
    picks = []
    i = 0
    while len(picks) < args.rows:
        b = avail[i % len(avail)]
        wshape = f.get_slice(b + ".weight").get().shape if hasattr(
            f.get_slice(b + ".weight"), "get") else None
        # shape via metadata-free probe: slice full (needed anyway)
        w = np.ascontiguousarray(f.get_slice(b + ".weight")[:])
        out_dim = w.shape[0]
        r = int(rng.integers(0, out_dim))
        picks.append((b, r))
        i += 1
        if i > args.rows * len(avail) + len(avail):
            break

    entries = []
    for b, r in picks:
        w = np.ascontiguousarray(f.get_slice(b + ".weight")[:])
        s = np.ascontiguousarray(f.get_slice(b + ".scales")[:]).astype(np.float64)
        n_in = w.shape[1] * 32
        words = w[r].astype(np.uint32)
        bits = np.unpackbits(words.view(np.uint8)).reshape(-1, 8)[:, ::-1].reshape(-1)[:n_in]
        sg = s[r] / 2.0
        wrow = np.where(bits == 1, np.repeat(sg, GROUP), -np.repeat(sg, GROUP))
        # unit-RMS random input mimics post-RMSNorm activations
        x = rng.standard_normal(n_in)
        x = x / np.sqrt(np.mean(x ** 2))
        y = float(np.dot(wrow, x))
        entries.append((b, r, n_in, x.astype(np.float32), y))

    with open(args.output, "wb") as fo:
        fo.write(MAGIC + struct.pack("<II", VERSION, len(entries)))
        for b, r, n_in, x, y in entries:
            nb = b.encode()
            fo.write(struct.pack("<H", len(nb)) + nb)
            fo.write(struct.pack("<II", r, n_in))
            fo.write(x.tobytes())
            fo.write(struct.pack("<d", y))
    print(f"wrote {args.output}: {len(entries)} vectors")


if __name__ == "__main__":
    main()
