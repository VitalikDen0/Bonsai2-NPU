#!/usr/bin/env python3
"""tokref_check: my BPE vs `tokenizers` lib reference on tokenizer.json.
Exit 0 iff all test strings encode identically.
"""
import json
import sys

try:
    import regex as re_mod
    HAVE_REGEX = True
except ImportError:
    import re as re_mod
    HAVE_REGEX = False

from tokenizers import Tokenizer

PACK = r"D:\Download\Bonsai2_NPU\Bonsai-27B-mlx-1bit"

# Qwen split pattern from tokenizer.json pre_tokenizer
SPLIT_PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
             r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

TESTS = [
    "Hello, world!",
    "Привет, как дела?",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "  leading and trailing  ",
    "x=1+2*3",
    "Ты — автономный локальный ИИ-агент.",
    "line1\nline2\n\nline3",
    "don't stop believin'",
    "a" * 50,
    "смешанный mixed текст 123 !@#",
]


def _bytes_to_unicode():
    # exact GPT-2 mapping
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

_B2U = _bytes_to_unicode()


def byte_encode_char(ch):
    # ByteLevel, no prefix space: raw UTF-8 bytes via GPT-2 table,
    # except ASCII space which Qwen writes as U+0120 (same as table: 32->U+0120).
    return [_B2U[b] for b in ch.encode("utf-8")]


def load_pack():
    tok = json.load(open(PACK + r"\tokenizer.json", encoding="utf-8"))
    vocab = tok["model"]["vocab"]          # token str -> id
    merges = tok["model"]["merges"]        # ["a b", ...] rank = index
    rank = {}
    for i, m in enumerate(merges):
        a, b = (m.split(" ") if isinstance(m, str) else (m[0], m[1]))
        rank[(a, b)] = i
    return vocab, rank


def bpe_piece(piece, vocab, rank):
    # piece: list of byte-level symbols; greedy lowest-rank merge
    toks = byte_encode_char(piece) if len(piece) == 1 and False else None
    # split piece into initial symbols (each char -> its byte symbols)
    syms = []
    for ch in piece:
        syms.extend(byte_encode_char(ch))
    while len(syms) > 1:
        best, bi = None, -1
        for i in range(len(syms) - 1):
            r = rank.get((syms[i], syms[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        syms = syms[:bi] + [syms[bi] + syms[bi + 1]] + syms[bi + 2:]
    ids = []
    for s in syms:
        if s not in vocab:
            return None
        ids.append(vocab[s])
    return ids


def my_encode(text, vocab, rank):
    ids = []
    for m in re_mod.findall(SPLIT_PAT, text):
        r = bpe_piece(m, vocab, rank)
        if r is None:
            return None
        ids.extend(r)
    return ids


def main():
    print("regex-module:", HAVE_REGEX)
    ref = Tokenizer.from_file(PACK + r"\tokenizer.json")
    vocab, rank = load_pack()
    fails = 0
    for t in TESTS:
        want = ref.encode(t).ids
        got = my_encode(t, vocab, rank)
        ok = (got == want)
        if not ok:
            fails += 1
            print(f"MISMATCH {t!r}\n  want={want}\n  got ={got}")
        else:
            print(f"OK len={len(want)} {t!r:.60}")
    print(f"tests={len(TESTS)} fails={fails}")
    print("TOKREF PASS" if fails == 0 else "TOKREF FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
