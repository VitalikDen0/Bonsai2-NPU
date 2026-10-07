// ref_gemv: bit-exact GATE runner for NPU1 containers (ARM64 reference, NOT product).
// Usage: ref_gemv <model.npubin> <gate.bin>
// Exit 0 + "GATE PASS" iff max abs err < 0.05 on all vectors.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0) f = s << 31;                       // subnormals -> 0 (scales never subnormal)
    else if (e == 31) f = (s << 31) | 0x7f800000 | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

struct Map { const uint8_t* p = nullptr; size_t n = 0;
    bool open(const char* path) {
        int fd = ::open(path, O_RDONLY); if (fd < 0) return false;
        struct stat st; if (fstat(fd, &st) != 0) return false;
        n = (size_t)st.st_size;
        p = (const uint8_t*)mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd); return p != MAP_FAILED;
    }
};

struct Reader {
    const uint8_t* p; size_t n; size_t o = 0;
    bool take(void* dst, size_t len) {
        if (o + len > n) return false;
        memcpy(dst, p + o, len); o += len; return true;
    }
    template<typename T> bool takev(T& v) { return take(&v, sizeof(v)); }
    bool str(std::string& s) {
        uint16_t l; if (!takev(l)) return false;
        if (o + l > n) return false;
        s.assign((const char*)(p + o), l); o += l; return true;
    }
};

struct Tensor { uint8_t kind; uint32_t out, in; uint64_t off, len; };

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: ref_gemv model.npubin gate.bin\n"); return 2; }
    Map m; if (!m.open(argv[1])) { fprintf(stderr, "open model failed\n"); return 2; }
    Reader r{m.p, m.n};
    if (memcmp(m.p, "NPU1", 4) != 0) { fprintf(stderr, "bad magic\n"); return 2; }
    r.o = 4;
    uint32_t ver, cnt; r.takev(ver); r.takev(cnt);
    if (ver != 0) { fprintf(stderr, "bad version\n"); return 2; }
    std::unordered_map<std::string, Tensor> tab;
    for (uint32_t i = 0; i < cnt; i++) {
        std::string name; Tensor t;
        uint16_t nl; r.takev(nl);
        name.assign((const char*)(m.p + r.o), nl); r.o += nl;
        r.takev(t.kind); r.takev(t.out); r.takev(t.in); r.takev(t.off); r.takev(t.len);
        tab[name] = t;
    }

    Map g; if (!g.open(argv[2])) { fprintf(stderr, "open gate failed\n"); return 2; }
    Reader gr{g.p, g.n};
    if (memcmp(g.p, "GATE1", 5) != 0) { fprintf(stderr, "bad gate magic\n"); return 2; }
    gr.o = 5;
    uint32_t gver, gcnt; gr.takev(gver); gr.takev(gcnt);
    double max_err = 0; int fails = 0;
    for (uint32_t i = 0; i < gcnt; i++) {
        std::string name; gr.str(name);
        uint32_t row, ind; gr.takev(row); gr.takev(ind);
        const float* x = (const float*)(g.p + gr.o); gr.o += (size_t)ind * 4;
        double yref; gr.takev(yref);
        auto it = tab.find(name);
        if (it == tab.end()) { printf("MISS %s\n", name.c_str()); fails++; continue; }
        const Tensor& t = it->second;
        if (t.kind != 0 || t.in != ind || row >= t.out) {
            printf("SHAPE %s\n", name.c_str()); fails++; continue;
        }
        uint32_t ng = ind / 128;
        uint32_t row_bytes = ind / 8;
        uint32_t prow = row_bytes + ((128 - (row_bytes % 128)) % 128);
        const uint16_t* scales = (const uint16_t*)(m.p + t.off);
        const uint8_t* bp = m.p + t.off + (size_t)t.out * ng * 2 + (size_t)row * prow;
        double y = 0;  // fp32 accumulation required by spec; double here only tightens
        float yf = 0;
        for (uint32_t j = 0; j < ind; j++) {
            int bit = (bp[j >> 3] >> (j & 7)) & 1;
            float sg = f16_to_f32(scales[(size_t)row * ng + j / 128]) * 0.5f;
            yf += (bit ? sg : -sg) * x[j];
        }
        y = yf;
        double err = fabs(y - yref);
        if (err > max_err) max_err = err;
        if (err >= 0.05) { printf("FAIL %s row %u err %.6f ref %.6f got %.6f\n",
                                  name.c_str(), row, err, yref, y); fails++; }
    }
    printf("vectors %u fails %d max_abs_err %.6f\n", gcnt, fails, max_err);
    printf(fails == 0 ? "GATE PASS\n" : "GATE FAIL\n");
    return fails == 0 ? 0 : 1;
}
