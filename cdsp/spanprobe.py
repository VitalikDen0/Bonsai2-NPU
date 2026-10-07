import regex as re_mod

pat = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
       r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
print("regex ver:", re_mod.__version__)
for s in ["  leading and trailing  ", "    return", "def fib(n):\n    return"]:
    print("TEXT:", repr(s))
    for m in re_mod.finditer(pat, s):
        print("   match:", repr(m.group(0)), m.span())
