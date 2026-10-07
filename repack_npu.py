#!/usr/bin/env python3
"""MLX 1-bit Bonsai 27B safetensors -> flat NPU container (v0, magic NPU1).

FORMAT (little-endian everywhere):
  header:  magic[4]=b'NPU1', u32 version=0, u32 tensor_count
  then tensor_count entries:
    u16 name_len, name bytes, u8 kind (0=binary-g128, 1=f16-raw),
    u32 out_dim, u32 in_dim, u64 data_offset, u64 data_len
  then data blobs (each 8-byte aligned):
    kind 0: scales fp16 (out*ngroups), then bitplane uint8 (out*padded_row_bytes)
            bit j of byte b of row r = weight (r, b*8+j); 1 -> +s_g, 0 -> -s_g
            s_g = stored_scale / 2  (MLX stores s_mlx = 2*s_g, bias = -s_g)
            padded_row_bytes = ceil((in_dim/8) / 128) * 128   (HVX 1024-bit rows)
            ngroups = in_dim / 128
    kind 1: raw fp16 row-major bytes, shape stored as out_dim=in_dim=0, dims in INDEX.

INDEX: separate JSON sidecar <output>.index.json mapping name -> {kind, shape}.
Bit order: lsb-first inside each uint32 word of the MLX pack
(word0.bit0 = weight0). Verified by self-test property |w| == s_g per group.
"""
import argparse
import json
import os
import struct
import sys
import tempfile

import numpy as np
from safetensors import safe_open

MAGIC = b"NPU1"
VERSION = 0
GROUP = 128
HVX_ROW = 128  # bytes, 1024-bit vector register

KIND_BIN = 0
KIND_F16 = 1


def open_st(path):
    return safe_open(path, framework="np")


def select_bases(keys, layers=None, include_vision=False):
    bases = {}
    for k in keys:
        if k.endswith(".weight") or k.endswith(".scales") or k.endswith(".biases"):
            b = k.rsplit(".", 1)[0]
            bases.setdefault(b, set()).add(k.rsplit(".", 1)[1])
    quant = sorted(b for b, parts in bases.items() if {"weight", "scales"} <= parts)
    covered = {q + s for q in quant for s in (".weight", ".scales", ".biases")}
    raw = sorted(k for k in keys if k not in covered)

    def layer_of(name):
        parts = name.split(".")
        if "layers" in parts:
            try:
                return int(parts[parts.index("layers") + 1])
            except ValueError:
                return None
        return None

    if not include_vision:
        quant = [b for b in quant if "language_model" in b]
        raw = [k for k in raw if "language_model" in k or "layers" not in k]
    if layers is not None:
        quant = [b for b in quant if layer_of(b) is None or layer_of(b) in layers]
        raw = [k for k in raw if layer_of(k) is None or layer_of(k) in layers]
    return quant, raw


def words_to_bits_lsb(words, n_in):
    """uint32 array -> 0/1 array (lsb-first: index i = bit i of word i//32)."""
    by = np.ascontiguousarray(words).view(np.uint8).reshape(-1)
    bits = np.unpackbits(by).reshape(-1, 8)[:, ::-1].reshape(-1)
    return bits[:n_in]


def bits_to_padded_row(bits, n_in):
    nb = n_in // 8
    assert n_in % 8 == 0
    # packbits is MSB-first: reverse each group of 8 to get lsb-first bytes
    pb = np.packbits(bits.reshape(-1, 8)[:, ::-1]).astype(np.uint8)
    assert len(pb) == nb
    pad = (-nb) % HVX_ROW
    if pad:
        pb = np.concatenate([pb, np.zeros(pad, dtype=np.uint8)])
    return pb


def build_blobs(f, quant, raw):
    """Returns list of (name, kind, out_dim, in_dim, shape, blob_bytes)."""
    out = []
    for b in quant:
        w = np.ascontiguousarray(f.get_slice(b + ".weight")[:])
        s = np.ascontiguousarray(f.get_slice(b + ".scales")[:]).astype(np.float16)
        out_dim, wcols = w.shape
        n_in = int(wcols) * 32
        ng = n_in // GROUP
        assert n_in % GROUP == 0, (b, w.shape)
        assert s.shape == (out_dim, ng), (b, s.shape)
        prow = (n_in // 8) + (-(n_in // 8) % HVX_ROW)
        row_bytes = n_in // 8
        # vectorized lsb-first bitplane, row chunks to bound RAM
        bp = np.empty((out_dim, prow), dtype=np.uint8)
        CH = 4096
        for c0 in range(0, out_dim, CH):
            c1 = min(out_dim, c0 + CH)
            wc = np.ascontiguousarray(w[c0:c1]).view(np.uint8).reshape(c1 - c0, -1)
            bits = np.unpackbits(wc).reshape(c1 - c0, -1, 8)[:, :, ::-1].reshape(c1 - c0, -1)
            bits = bits[:, :n_in]
            bp[c0:c1, :row_bytes] = np.packbits(
                bits.reshape(c1 - c0, -1, 8)[:, :, ::-1], axis=-1).reshape(c1 - c0, row_bytes)
            if prow > row_bytes:
                bp[c0:c1, row_bytes:] = 0
        blob = np.ascontiguousarray(s).tobytes() + bp.tobytes()
        out.append((b, KIND_BIN, out_dim, n_in, [out_dim, n_in], blob))
    for k in raw:
        a = np.ascontiguousarray(f.get_slice(k)[:]).astype(np.float16)
        shape = list(a.shape)
        od = shape[0] if len(shape) >= 1 else 1
        ind = int(np.prod(shape[1:])) if len(shape) >= 2 else 1
        out.append((k, KIND_F16, od, ind, shape, a.tobytes()))
    return out


def write_container(entries, out_path):
    header = MAGIC + struct.pack("<II", VERSION, len(entries))
    table = b""
    for (name, kind, od, ind, shape, blob) in entries:
        nb = name.encode()
        table += struct.pack("<H", len(nb)) + nb
        table += struct.pack("<BII", kind, od, ind)
        table += struct.pack("<QQ", 0, len(blob))  # offset patched below
    header += table
    pad = (-len(header)) % 8
    header += b"\x00" * pad
    # patch offsets
    off = len(header)
    header = bytearray(header)
    p = 4 + 8
    for (name, kind, od, ind, shape, blob) in entries:
        nl = struct.unpack_from("<H", header, p)[0]
        p += 2 + nl + 9
        struct.pack_into("<Q", header, p, off)
        p += 16
        off += len(blob)
        off += (-len(blob)) % 8
    with open(out_path, "wb") as fo:
        fo.write(header)
        for (name, kind, od, ind, shape, blob) in entries:
            fo.write(blob)
            m = (-len(blob)) % 8
            if m:
                fo.write(b"\x00" * m)
    index = {name: {"kind": kind, "shape": shape} for (name, kind, od, ind, shape, blob) in entries}
    with open(out_path + ".index.json", "w") as fo:
        json.dump(index, fo)


def read_container(path):
    with open(path, "rb") as fo:
        data = fo.read()
    assert data[:4] == MAGIC
    ver, n = struct.unpack_from("<II", data, 4)
    assert ver == 0
    p = 12
    entries = []
    for _ in range(n):
        nl = struct.unpack_from("<H", data, p)[0]
        p += 2
        name = data[p:p + nl].decode()
        p += nl
        kind, od, ind = struct.unpack_from("<BII", data, p)
        p += 9
        doff, blen = struct.unpack_from("<QQ", data, p)
        p += 16
        entries.append((name, kind, od, ind, doff, blen))
    return entries, data


def self_test(st_path):
    f = open_st(st_path)
    quant, raw = select_bases(list(f.keys()), layers={0, 1})
    print(f"self-test: {len(quant)} quant + {len(raw)} raw tensors (layers 0,1)")
    assert len(quant) > 0 and len(raw) > 0
    tmp = tempfile.mktemp(suffix=".npubin")
    entries = build_blobs(f, quant, raw)
    write_container(entries, tmp)
    back, data = read_container(tmp)
    emap = {e[0]: e for e in back}
    max_err = 0.0
    checked = 0
    rng = np.random.default_rng(0)
    for b in quant:
        w = np.ascontiguousarray(f.get_slice(b + ".weight")[:])
        s = np.ascontiguousarray(f.get_slice(b + ".scales")[:]).astype(np.float32)
        out_dim, wcols = w.shape
        n_in = int(wcols) * 32
        name, kind, od, ind, doff, blen = emap[b]
        ng = n_in // GROUP
        assert (od, ind) == (out_dim, n_in)
        for r in rng.choice(out_dim, size=min(3, out_dim), replace=False):
            r = int(r)
            bits = words_to_bits_lsb(w[r], n_in)
            sg = s[r] / 2.0
            ref = np.where(bits == 1, np.repeat(sg, GROUP), -np.repeat(sg, GROUP))
            # magnitude property: |w| == s_g in every group
            assert np.all(np.abs(ref).reshape(-1, GROUP)
                          == np.repeat(sg, GROUP).reshape(-1, GROUP)), f"magprop {b} r{r}"
            scales = np.frombuffer(data, dtype=np.float16, count=od * ng, offset=doff
                                   ).reshape(od, ng).astype(np.float32)
            assert np.array_equal(scales[r], s[r]), f"scale mismatch {b} r{r}"
            prow = (blen - od * ng * 2) // od
            nb = n_in // 8
            rowb = np.frombuffer(data, dtype=np.uint8, count=nb, offset=doff + od * ng * 2 + r * prow)
            b2 = np.unpackbits(rowb).reshape(-1, 8)[:, ::-1].reshape(-1)[:n_in]
            got = np.where(b2 == 1, np.repeat(sg, GROUP), -np.repeat(sg, GROUP))
            err = float(np.max(np.abs(got - ref)))
            max_err = max(max_err, err)
            checked += 1
    for k in raw:
        a = np.ascontiguousarray(f.get_slice(k)[:]).astype(np.float16)
        name, kind, od, ind, doff, blen = emap[k]
        back_a = np.frombuffer(data, dtype=np.float16, count=a.size, offset=doff).reshape(a.shape)
        assert np.array_equal(a, back_a), f"raw mismatch {k}"
        checked += 1
    os.remove(tmp)
    os.remove(tmp + ".index.json")
    print(f"self-test: {checked} rows/tensors checked, max abs err = {max_err}")
    assert max_err == 0.0
    print("SELF-TEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False)
    ap.add_argument("--output", required=False)
    ap.add_argument("--layers", default=None, help="e.g. 0,1")
    ap.add_argument("--include-vision", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        if not args.input:
            print("--self-test needs --input", file=sys.stderr)
            sys.exit(2)
        self_test(args.input)
        return
    if not args.input or not args.output:
        print("need --input and --output (or --self-test --input)", file=sys.stderr)
        sys.exit(2)
    layers = {int(x) for x in args.layers.split(",")} if args.layers else None
    f = open_st(args.input)
    quant, raw = select_bases(list(f.keys()), layers=layers, include_vision=args.include_vision)
    print(f"packing {len(quant)} quant + {len(raw)} raw tensors")
    write_container(build_blobs(f, quant, raw), args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
