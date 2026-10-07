// tok: byte-level BPE encode/decode for Bonsai 27B (Qwen split pattern).
// Algorithm validated in Python vs `tokenizers` lib (TOKREF PASS 10/10).
// Self-test: ./tok_selftest tok.bin  (checks embedded vectors, no device needed
// to build; runs on device).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const unsigned char UCAT[1114112];  // bit0=L bit1=M bit2=N (ucat.c)

// ---------- byte table (exact GPT-2 bytes_to_unicode) ----------
static int b2u_init_done = 0;
static unsigned b2u_tab[256];   // byte -> unicode scalar
static int u2b_tab[0x300];      // scalar (<0x300) -> byte, -1 unknown
static void b2u_init(void) {
    if (b2u_init_done) return;
    int in_print[256] = {0};
    for (int b = 0x21; b <= 0x7E; b++) in_print[b] = 1;
    for (int b = 0xA1; b <= 0xAC; b++) in_print[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++) in_print[b] = 1;
    for (int i = 0; i < 0x300; i++) u2b_tab[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        unsigned cp = in_print[b] ? (unsigned)b : (unsigned)(256 + n++);
        b2u_tab[b] = cp;
        if (cp < 0x300) u2b_tab[cp] = b;
    }
    b2u_init_done = 1;
}

// ---------- tok.bin ----------
typedef struct { char* s; int len; } tstr;
typedef struct { uint32_t l, r; } mpair;

static tstr* vocab_strs;
static int vocab_n;
static mpair* merges;      // by rank
static int merges_n;
static tstr* spec_strs;
static uint32_t* spec_ids;
static int spec_n;

// FNV-1a hash table: token bytes -> id
typedef struct { const char* s; int len; int id; int used; } hent;
static hent* htab;
static int htab_n;

static unsigned fnv(const char* s, int len) {
    unsigned h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static const uint8_t* rp;
static const uint8_t* rend;
static int rerr = 0;
static void rb(void* d, int n) {
    if (rp + n > rend) { rerr = 1; return; }
    memcpy(d, rp, (size_t)n); rp += n;
}
static uint16_t ru16(void) { uint16_t v = 0; rb(&v, 2); return v; }
static uint32_t ru32(void) { uint32_t v = 0; rb(&v, 4); return v; }

static int load_tok(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return -1; }
    fclose(f);
    rp = buf; rend = buf + sz;
    if (memcmp(rp, "TOK1", 4) != 0) return -1;
    rp += 4;
    uint32_t ver = ru32();
    int nv = (int)ru32(), nm = (int)ru32(), ns = (int)ru32();
    int nvec = (int)ru32();
    (void)nvec;
    if (ver != 0 || rerr) return -1;
    vocab_n = nv; merges_n = nm; spec_n = ns;
    vocab_strs = (tstr*)calloc((size_t)nv, sizeof(tstr));
    for (int i = 0; i < nv; i++) {
        int L = ru16();
        vocab_strs[i].s = (char*)rp; vocab_strs[i].len = L; rp += L;
    }
    merges = (mpair*)malloc(sizeof(mpair) * (size_t)nm);
    for (int i = 0; i < nm; i++) { merges[i].l = ru32(); merges[i].r = ru32(); }
    spec_strs = (tstr*)calloc((size_t)(ns > 0 ? ns : 1), sizeof(tstr));
    spec_ids = (uint32_t*)calloc((size_t)(ns > 0 ? ns : 1), sizeof(uint32_t));
    for (int i = 0; i < ns; i++) {
        int L = ru16();
        spec_strs[i].s = (char*)rp; spec_strs[i].len = L; rp += L;
        spec_ids[i] = ru32();
    }
    if (rerr) return -1;
    // hash table 2x
    htab_n = 1;
    while (htab_n < nv * 2) htab_n <<= 1;
    htab = (hent*)calloc((size_t)htab_n, sizeof(hent));
    for (int i = 0; i < nv; i++) {
        unsigned h = fnv(vocab_strs[i].s, vocab_strs[i].len) & (unsigned)(htab_n - 1);
        while (htab[h].used) h = (h + 1) & (unsigned)(htab_n - 1);
        htab[h].used = 1;
        htab[h].s = vocab_strs[i].s;
        htab[h].len = vocab_strs[i].len;
        htab[h].id = i;
    }
    return 0;
}

static int vocab_lookup(const char* s, int len) {
    unsigned h = fnv(s, len) & (unsigned)(htab_n - 1);
    while (htab[h].used) {
        if (htab[h].len == len && memcmp(htab[h].s, s, (size_t)len) == 0) return htab[h].id;
        h = (h + 1) & (unsigned)(htab_n - 1);
    }
    return -1;
}

// merge pairs sorted for binary search -> rank
typedef struct { uint32_t l, r; int rank; } spair;
static spair* spairs;
static int spair_cmp(const void* a, const void* b) {
    const spair* x = (const spair*)a, *y = (const spair*)b;
    if (x->l != y->l) return x->l < y->l ? -1 : 1;
    if (x->r != y->r) return x->r < y->r ? -1 : 1;
    return 0;
}
static void build_spairs(void) {
    spairs = (spair*)malloc(sizeof(spair) * (size_t)merges_n);
    for (int i = 0; i < merges_n; i++) { spairs[i].l = merges[i].l; spairs[i].r = merges[i].r; spairs[i].rank = i; }
    qsort(spairs, (size_t)merges_n, sizeof(spair), spair_cmp);
}
static int pair_rank(uint32_t l, uint32_t r) {
    int lo = 0, hi = merges_n - 1;
    while (lo <= hi) {
        int m = (lo + hi) >> 1;
        if (spairs[m].l == l && spairs[m].r == r) return spairs[m].rank;
        if (spairs[m].l < l || (spairs[m].l == l && spairs[m].r < r)) lo = m + 1;
        else hi = m - 1;
    }
    return -1;
}

// ---------- UTF-8 decode ----------
static int utf8_dec(const unsigned char* s, int n, unsigned* cp) {
    if (n <= 0) return -1;
    unsigned char c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && n >= 2) { *cp = ((unsigned)(c & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0 && n >= 3) { *cp = ((unsigned)(c & 0x0F) << 12) | ((unsigned)(s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
    if ((c & 0xF8) == 0xF0 && n >= 4) { *cp = ((unsigned)(c & 0x07) << 18) | ((unsigned)(s[1] & 0x3F) << 12) | ((unsigned)(s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4; }
    return -1;
}
static int is_space(unsigned cp) { return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D; }
static int is_nl(unsigned cp) { return cp == 0x0A || cp == 0x0D; }

#define MAXPIECE 4096

// BPE merge loop over symbol ids; returns count or -1
static int bpe_merge(int* syms, int n) {
    // work with dynamic concatenation per merge (prompt-time speed OK)
    int* cur = (int*)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    char** str = (char**)malloc(sizeof(char*) * (size_t)(n > 0 ? n : 1));
    int* slen = (int*)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; i++) {
        cur[i] = syms[i];
        slen[i] = vocab_strs[syms[i]].len;
        str[i] = (char*)malloc((size_t)slen[i] + 1);
        memcpy(str[i], vocab_strs[syms[i]].s, (size_t)slen[i]);
    }
    int m = n;
    static char cat[8192];
    while (m > 1) {
        int best = -1, bi = -1;
        for (int i = 0; i < m - 1; i++) {
            int r = pair_rank((uint32_t)cur[i], (uint32_t)cur[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (best < 0) break;
        int L = slen[bi] + slen[bi + 1];
        if (L >= (int)sizeof(cat)) { m = -1; break; }
        memcpy(cat, str[bi], (size_t)slen[bi]);
        memcpy(cat + slen[bi], str[bi + 1], (size_t)slen[bi + 1]);
        int nid = vocab_lookup(cat, L);
        if (nid < 0) { m = -1; break; }
        free(str[bi]);
        free(str[bi + 1]);
        str[bi] = (char*)malloc((size_t)L + 1);
        memcpy(str[bi], cat, (size_t)L);
        slen[bi] = L;
        cur[bi] = nid;
        for (int i = bi + 1; i < m - 1; i++) { cur[i] = cur[i + 1]; str[i] = str[i + 1]; slen[i] = slen[i + 1]; }
        m--;
    }
    if (m < 0) { for (int i = 0; i < n; i++) free(str[i]); free(cur); free(str); free(slen); return -1; }
    for (int i = 0; i < m; i++) { syms[i] = cur[i]; free(str[i]); }
    free(cur); free(str); free(slen);
    return m;
}

// Qwen pre-split: emit pieces as (start,end) codepoint ranges into out[][2]; returns npieces.
// piece buffer cps[MAXTEXT] shared.
#define MAXTEXT 16384
static unsigned g_cps[MAXTEXT];
static int g_piece_s[MAXTEXT], g_piece_e[MAXTEXT];

static int try_contraction(const unsigned* cps, int pos, int ncp, int* len) {
    // (?i:'s|'t|'re|'ve|'m|'ll|'d) with optional leading ' handled by caller piece start
    if (cps[pos] != '\'') return 0;
    // 's 't 'm 'd 'll 're 've  (case-insensitive first letter after ')
    if (pos + 1 >= ncp) return 0;
    unsigned c1 = cps[pos + 1];
    unsigned l1 = (c1 >= 'A' && c1 <= 'Z') ? c1 + 32 : c1;
    if (l1 == 's' || l1 == 't' || l1 == 'm' || l1 == 'd') { *len = 2; return 1; }
    if (pos + 2 < ncp) {
        unsigned c2 = cps[pos + 2];
        unsigned l2 = (c2 >= 'A' && c2 <= 'Z') ? c2 + 32 : c2;
        if ((l1 == 'r' && l2 == 'e') || (l1 == 'v' && l2 == 'e') || (l1 == 'l' && l2 == 'l')) { *len = 3; return 1; }
    }
    return 0;
}

static int qwen_split(const unsigned char* text, int nbytes) {
    int ncp = 0;
    int p = 0;
    while (p < nbytes && ncp < MAXTEXT) {
        unsigned cp;
        int L = utf8_dec(text + p, nbytes - p, &cp);
        if (L < 0) return -1;
        g_cps[ncp++] = cp;
        p += L;
    }
    int np = 0, i = 0;
    while (i < ncp) {
        unsigned cp = g_cps[i];
        int ucat = (cp < 1114112u) ? (UCAT[cp] & 7) : 0;
        int isL = (ucat & 1) != 0, isM = (ucat & 2) != 0, isN = (ucat & 4) != 0;
        int isLMN = (ucat & 7) != 0;
        // alt 1: contraction starting with '
        if (cp == '\'') {
            int L = 0;
            if (try_contraction(g_cps, i, ncp, &L)) {
                g_piece_s[np] = i; g_piece_e[np] = i + L; np++;
                i += L;
                continue;
            }
        }
        // alt 2: [^\r\nLN]?[LM]+
        {
            int j = (is_nl(cp) || isLMN) ? i : i + 1;
            int k = j;
            while (k < ncp) {
                unsigned c2 = g_cps[k];
                int u2 = (c2 < 1114112u) ? (UCAT[c2] & 7) : 0;
                if (!((u2 & 1) || (u2 & 2))) break;
                k++;
            }
            if (k > j) {
                g_piece_s[np] = i; g_piece_e[np] = k; np++;
                i = k;
                continue;
            }
        }
        // alt 3: single \p{N}
        if (isN) { g_piece_s[np] = i; g_piece_e[np] = i + 1; np++; i++; continue; }
        // alt 4: ' ?[^\sLMN]+[\r\n]*'
        {
            int j = (cp == 0x20) ? i + 1 : i;
            int k = j;
            while (k < ncp) {
                unsigned c2 = g_cps[k];
                if (is_space(c2)) break;
                int u2 = (c2 < 1114112u) ? (UCAT[c2] & 7) : 0;
                if (u2 & 7) break;
                k++;
            }
            if (k > j) {
                while (k < ncp && is_nl(g_cps[k])) k++;
                g_piece_s[np] = i; g_piece_e[np] = k; np++;
                i = k;
                continue;
            }
        }
        // alt 5: \s*[\r\n]+ — maximal whitespace run, cut after its LAST newline
        // (greedy \s* + backtrack semantics)
        {
            int j = i;
            while (j < ncp && is_space(g_cps[j])) j++;
            int last_nl = -1;
            for (int t = i; t < j; t++) {
                if (is_nl(g_cps[t])) last_nl = t;
            }
            if (last_nl >= 0) {
                g_piece_s[np] = i; g_piece_e[np] = last_nl + 1; np++;
                i = last_nl + 1;
                continue;
            }
        }
        // alt 6: \s+(?!\S) — space run (no newlines: alt5 handled those);
        // greedy take, then require next-is-space-or-end with one-char backtrack
        {
            int j = i;
            while (j < ncp && is_space(g_cps[j]) && !is_nl(g_cps[j])) j++;
            int k = j - i;
            if (k > 0) {
                if (j >= ncp) {
                    g_piece_s[np] = i; g_piece_e[np] = j; np++;
                    i = j;
                    continue;
                }
                if (k > 1) {
                    g_piece_s[np] = i; g_piece_e[np] = j - 1; np++;
                    i = j - 1;
                    continue;
                }
                // k == 1 before a word char: alt6 fails -> alt7 below
            }
        }
        // alt 7: \s+
        {
            int j = i;
            while (j < ncp && is_space(g_cps[j]) && !is_nl(g_cps[j])) j++;
            if (j > i) {
                g_piece_s[np] = i; g_piece_e[np] = j; np++;
                i = j;
                continue;
            }
        }
        // fallback (unreachable for valid input): single codepoint
        g_piece_s[np] = i; g_piece_e[np] = i + 1; np++;
        i++;
    }
    return np;
}

// encode text -> ids; returns count or -1. Specials scanned longest-first.
typedef struct { tstr s; uint32_t id; } sspec;
static sspec* sspecs;

static int tok_encode(const unsigned char* text, int nbytes, int* out, int cap) {
    int nout = 0;
    int p = 0;
    while (p < nbytes) {
        // longest special match at p
        int best = -1, bestlen = 0;
        for (int i = 0; i < spec_n; i++) {
            int L = sspecs[i].s.len;
            if (L > bestlen && p + L <= nbytes && memcmp(text + p, sspecs[i].s.s, (size_t)L) == 0) {
                best = i; bestlen = L;
            }
        }
        if (best >= 0) {
            if (nout >= cap) return -1;
            out[nout++] = (int)sspecs[best].id;
            p += bestlen;
            continue;
        }
        // run until next special
        int q = p + 1;
        while (q < nbytes) {
            int hit = 0;
            for (int i = 0; i < spec_n; i++) {
                int L = sspecs[i].s.len;
                if (q + L <= nbytes && memcmp(text + q, sspecs[i].s.s, (size_t)L) == 0) { hit = 1; break; }
            }
            if (hit) break;
            q++;
        }
        int np = qwen_split(text + p, q - p);
        if (np < 0) return -1;
        // pieces are codepoint ranges of the SUBSTRING: need codepoints; re-decode substring
        // qwen_split filled g_cps for substring; pieces index into it.
        for (int k = 0; k < np; k++) {
            int a = g_piece_s[k], b = g_piece_e[k];
            // byte-encode codepoints g_cps[a..b)
            static int syms[MAXPIECE];
            int ns = 0;
            for (int i = a; i < b; i++) {
                unsigned cp = g_cps[i];
                unsigned char ub[4];
                int ul = 0;
                if (cp < 0x80) ub[ul++] = (unsigned char)cp;
                else if (cp < 0x800) { ub[ul++] = (unsigned char)(0xC0 | (cp >> 6)); ub[ul++] = (unsigned char)(0x80 | (cp & 0x3F)); }
                else if (cp < 0x10000) { ub[ul++] = (unsigned char)(0xE0 | (cp >> 12)); ub[ul++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F)); ub[ul++] = (unsigned char)(0x80 | (cp & 0x3F)); }
                else { ub[ul++] = (unsigned char)(0xF0 | (cp >> 18)); ub[ul++] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F)); ub[ul++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F)); ub[ul++] = (unsigned char)(0x80 | (cp & 0x3F)); }
                for (int t2 = 0; t2 < ul; t2++) {
                    unsigned sc = b2u_tab[ub[t2]];
                    char tmp[4];
                    int tl = 0;
                    if (sc < 0x80) tmp[tl++] = (char)sc;
                    else if (sc < 0x800) { tmp[tl++] = (char)(0xC0 | (sc >> 6)); tmp[tl++] = (char)(0x80 | (sc & 0x3F)); }
                    else { tmp[tl++] = (char)(0xE0 | (sc >> 12)); tmp[tl++] = (char)(0x80 | ((sc >> 6) & 0x3F)); tmp[tl++] = (char)(0x80 | (sc & 0x3F)); }
                    int id = vocab_lookup(tmp, tl);
                    if (id < 0 || ns >= MAXPIECE) return -1;
                    syms[ns++] = id;
                }
            }
            int m = bpe_merge(syms, ns);
            if (m < 0 || nout + m > cap) return -1;
            for (int i = 0; i < m; i++) out[nout++] = syms[i];
        }
        p = q;
    }
    return nout;
}

// decode ids -> bytes; returns byte count or -1.
static int tok_decode(const int* ids, int nids, unsigned char* out, int cap) {
    int n = 0;
    for (int k = 0; k < nids; k++) {
        int id = ids[k];
        const char* s;
        int L;
        // special?
        int is_spec = 0;
        for (int i = 0; i < spec_n; i++) {
            if ((int)spec_ids[i] == id) { s = spec_strs[i].s; L = spec_strs[i].len; is_spec = 1; break; }
        }
        if (!is_spec) {
            if (id < 0 || id >= vocab_n) return -1;
            s = vocab_strs[id].s; L = vocab_strs[id].len;
        }
        // map token string chars back to bytes
        int p = 0;
        while (p < L) {
            unsigned cp;
            int dl = utf8_dec((const unsigned char*)s + p, L - p, &cp);
            if (dl < 0) return -1;
            int b;
            if (cp < 0x300) b = u2b_tab[cp];
            else b = -1;
            if (!is_spec && b < 0) return -1;
            if (is_spec) {
                if (n + dl > cap) return -1;
                memcpy(out + n, s + p, (size_t)dl);
                n += dl;
            } else {
                if (n + 1 > cap) return -1;
                out[n++] = (unsigned char)b;
            }
            p += dl;
        }
    }
    return n;
}

#ifdef TOK_SELFTEST
// vectors appended after specials in tok.bin by gen_tok.py
static int run_vectors(const uint8_t* base, const uint8_t* end) {
    // re-walk tok.bin to vector section: recompute offsets
    const uint8_t* p = base;
    if (memcmp(p, "TOK1", 4) != 0) return -1;
    p += 4;
    uint32_t ver, nv, nm, ns, nvec;
    memcpy(&ver, p, 4); p += 4;
    memcpy(&nv, p, 4); p += 4;
    memcpy(&nm, p, 4); p += 4;
    memcpy(&ns, p, 4); p += 4;
    memcpy(&nvec, p, 4); p += 4;
    for (uint32_t i = 0; i < nv; i++) { uint16_t L; memcpy(&L, p, 2); p += 2 + L; }
    p += (size_t)nm * 8;
    for (uint32_t i = 0; i < ns; i++) { uint16_t L; memcpy(&L, p, 2); p += 2 + L + 4; }
    static int ids[32768];
    static unsigned char dec[65536];
    int fails = 0;
    for (uint32_t v = 0; v < nvec; v++) {
        uint16_t tl;
        memcpy(&tl, p, 2); p += 2;
        const unsigned char* txt = p; p += tl;
        uint32_t ni;
        memcpy(&ni, p, 4); p += 4;
        const uint32_t* want = (const uint32_t*)p; p += (size_t)ni * 4;
        int got = tok_encode(txt, tl, ids, 32768);
        if (got != (int)ni) { printf("VEC %u COUNT got=%d want=%u\n", v, got, ni); fails++; continue; }
        int bad = 0;
        for (uint32_t i = 0; i < ni; i++) if (ids[i] != (int)want[i]) bad = 1;
        if (bad) { printf("VEC %u IDS differ\n", v); fails++; continue; }
        int nb = tok_decode(ids, got, dec, 65536);
        if (nb != tl || memcmp(dec, txt, (size_t)tl) != 0) { printf("VEC %u ROUNDTRIP\n", v); fails++; continue; }
        printf("VEC %u ok nids=%u\n", v, ni);
    }
    printf(fails == 0 ? "TOK SELFTEST PASS\n" : "TOK SELFTEST FAIL\n");
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: tok_selftest tok.bin\n"); return 2; }
    b2u_init();
    if (load_tok(argv[1]) != 0) { printf("LOAD FAIL\n"); return 2; }
    build_spairs();
    sspecs = (sspec*)calloc((size_t)(spec_n > 0 ? spec_n : 1), sizeof(sspec));
    for (int i = 0; i < spec_n; i++) { sspecs[i].s = spec_strs[i]; sspecs[i].id = spec_ids[i]; }
    FILE* f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) return 2;
    fclose(f);
    return run_vectors(buf, buf + sz);
}
#endif
