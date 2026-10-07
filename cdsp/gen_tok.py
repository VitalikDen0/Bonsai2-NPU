#!/usr/bin/env python3
"""gen_tok: pack tokenizer.json into compact tok.bin + test vectors.

tok.bin (little-endian):
  magic[4]=b'TOK1', u32 version=0, u32 vocab_size,
  u32 n_merges, u32 n_specials, u32 n_vectors,
  then vocab_size entries: u16 bytelen, bytes (utf-8 token string),
  then n_merges entries: u32 left_id, u32 right_id
    (merge rank = position; left/right are single-symbol token ids;
     multi-char merges resolved at runtime by id lookup),
  then n_specials entries: u16 bytelen, bytes, u32 id,
  then vectors: u32 n, then n cases: u16 textlen, text bytes,
    u32 nids, u32 ids[nids].
"""
import argparse
import json
import struct

MAGIC = b"TOK1"
VERSION = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="tokenizer.json dir (tokenizer.json inside)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vectors", default=64)
    args = ap.parse_args()

    tok = json.load(open(args.pack + r"\tokenizer.json", encoding="utf-8"))
    vocab = tok["model"]["vocab"]          # str -> id
    merges = tok["model"]["merges"]        # ["a b"] or [["a","b"]]
    added = tok.get("added_tokens", [])

    id2s = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in id2s] == list(range(len(id2s))), "vocab ids not dense"
    strs = [s for s, _ in id2s]

    mids = []
    for m in merges:
        a, b = (m.split(" ") if isinstance(m, str) else (m[0], m[1]))
        mids.append((vocab[a], vocab[b]))

    specs = [(a["content"], a["id"]) for a in added]

    # test vectors from tokref_check corpus + specials roundtrip
    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(args.pack + r"\tokenizer.json")
    corpus = [
        "Hello, world!", "Привет, как дела?",
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
        "  leading and trailing  ", "x=1+2*3",
        "Ты — автономный локальный ИИ-агент.",
        "line1\nline2\n\nline3", "don't stop believin'",
        "a" * 50, "смешанный mixed текст 123 !@#",
        "<|im_start|>system<|im_end|><|im_start|>user<|im_end|>",
        "HumanEval: assert add(2, 3) == 5",
    ]
    vecs = []
    for t in corpus[:args.vectors]:
        ids = ref.encode(t).ids
        vecs.append((t, ids))

    with open(args.output, "wb") as fo:
        fo.write(MAGIC + struct.pack("<IIII", VERSION, len(strs), len(mids), len(specs)))
        fo.write(struct.pack("<I", len(vecs)))
        for s in strs:
            b = s.encode("utf-8")
            fo.write(struct.pack("<H", len(b)) + b)
        for a, b in mids:
            fo.write(struct.pack("<II", a, b))
        for s, i in specs:
            b = s.encode("utf-8")
            fo.write(struct.pack("<H", len(b)) + b)
            fo.write(struct.pack("<I", i))
        for t, ids in vecs:
            b = t.encode("utf-8")
            fo.write(struct.pack("<H", len(b)) + b)
            fo.write(struct.pack("<I", len(ids)))
            fo.write(struct.pack("<%dI" % len(ids), *ids))
    print(f"wrote {args.output}: vocab={len(strs)} merges={len(mids)} "
          f"specials={len(specs)} vectors={len(vecs)}")


if __name__ == "__main__":
    main()
