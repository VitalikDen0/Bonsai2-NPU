#!/usr/bin/env python3
"""Replicate tok.c qwen_split EXACTLY in Python; diff pieces vs reference encode."""
import sys
sys.path.insert(0, r"D:\Download\Bonsai2_NPU\cdsp")
from tokref_check import load_pack, SPLIT_PAT
import regex as re_mod
from tokenizers import Tokenizer

PACK = r"D:\Download\Bonsai2_NPU\Bonsai-27B-mlx-1bit"
vocab, rank = load_pack()

CAT = {}
for cp in range(0x110000):
    if 0xD800 <= cp <= 0xDFFF:
        continue
    ch = chr(cp)
    v = 0
    if re_mod.match(r"\p{L}", ch):
        v |= 1
    if re_mod.match(r"\p{M}", ch):
        v |= 2
    if re_mod.match(r"\p{N}", ch):
        v |= 4
    if v:
        CAT[cp] = v


def is_space(cp):
    return cp in (0x20, 0x09, 0x0A, 0x0D)


def is_nl(cp):
    return cp in (0x0A, 0x0D)


def try_contr(cps, pos):
    if cps[pos] != 0x27:
        return 0
    if pos + 1 >= len(cps):
        return 0
    c1 = cps[pos + 1]
    l1 = c1 + 32 if 65 <= c1 <= 90 else c1
    if l1 in (ord('s'), ord('t'), ord('m'), ord('d')):
        return 2
    if pos + 2 < len(cps):
        c2 = cps[pos + 2]
        l2 = c2 + 32 if 65 <= c2 <= 90 else c2
        if (l1 == ord('r') and l2 == ord('e')) or \
           (l1 == ord('v') and l2 == ord('e')) or \
           (l1 == ord('l') and l2 == ord('l')):
            return 3
    return 0


def c_split(text):
    cps = [ord(c) for c in text]
    ncp = len(cps)
    out = []
    i = 0
    while i < ncp:
        cp = cps[i]
        ucat = CAT.get(cp, 0)
        isL = bool(ucat & 1)
        isM = bool(ucat & 2)
        isN = bool(ucat & 4)
        isLMN = bool(ucat & 7)
        if cp == 0x27:
            L = try_contr(cps, i)
            if L:
                out.append((i, i + L))
                i += L
                continue
        # alt 2
        j = i if (is_nl(cp) or isLMN) else i + 1
        k = j
        while k < ncp:
            u2 = CAT.get(cps[k], 0)
            if not (u2 & 1 or u2 & 2):
                break
            k += 1
        if k > j:
            out.append((i, k))
            i = k
            continue
        if isN:
            out.append((i, i + 1))
            i += 1
            continue
        # alt 4
        j = i + 1 if cp == 0x20 else i
        k = j
        while k < ncp:
            c2 = cps[k]
            if is_space(c2):
                break
            if CAT.get(c2, 0) & 7:
                break
            k += 1
        if k > j:
            while k < ncp and is_nl(cps[k]):
                k += 1
            out.append((i, k))
            i = k
            continue
        # alt 5: \s*[\r\n]+ — maximal whitespace run, cut after its LAST newline
        j = i
        while j < ncp and is_space(cps[j]):
            j += 1
        last_nl = -1
        for t in range(i, j):
            if is_nl(cps[t]):
                last_nl = t
        if last_nl >= 0:
            out.append((i, last_nl + 1))
            i = last_nl + 1
            continue
        # alt 6: \s+(?!\S) — space run; require next-is-space-or-end w/ backtrack
        j = i
        while j < ncp and is_space(cps[j]) and not is_nl(cps[j]):
            j += 1
        k = j - i
        if k > 0:
            if j >= ncp:
                out.append((i, j))
                i = j
                continue
            if k > 1:
                out.append((i, j - 1))
                i = j - 1
                continue
            # k == 1 before word char: alt6 fails -> alt7
        # alt 7: \s+
        j = i
        while j < ncp and is_space(cps[j]) and not is_nl(cps[j]):
            j += 1
        if j > i:
            out.append((i, j))
            i = j
            continue
        out.append((i, i + 1))
        i += 1
    return [text[a:b] for a, b in out]


ref = Tokenizer.from_file(PACK + r"\tokenizer.json")
tests = [
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "  leading and trailing  ",
    "Hello, world!",
    "Привет, как дела?",
    "  leading",
    "trailing  ",
    "a  b   c",
    "\n\n\n",
    "  \n  \n",
    "x\n\ny",
    "   \nhello",
    "hello\n   world",
    " a ",
    "\tindented\t",
    "don't stop",
    "'tis 'Twas 'RE 'Ve 'M 'LL 'D 'x",
    "123 456",
    "a-b-c",
    "-abc 5abc",
    "word.",
    "end...",
    "mix\tof  spaces\nand\nnewlines  ",
    "Ты — автономный, локальный ИИ-агент!",
]
for t in tests:
    mine = c_split(t)
    # reference pieces via regex findall
    want = re_mod.findall(SPLIT_PAT, t)
    print("TEXT:", repr(t[:40]))
    print("  C-port pieces:", mine)
    print("  ref   pieces:", want)
    print("  match:", mine == want)
