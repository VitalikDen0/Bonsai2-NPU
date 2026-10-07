import regex as re_mod

s = "  leading"
alts = [
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
]
for i, a in enumerate(alts):
    ms = [(m.group(0), m.span()) for m in re_mod.finditer(a, s)]
    print(i, repr(a[:30]), "->", ms[:4])
