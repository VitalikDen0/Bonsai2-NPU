// gate_npu: run GATE1 vectors through the REAL CDSP module via generated stub.
// Usage: gate_npu <model.npubin> <gate.bin>
// Exit 0 + "NPU GATE PASS" iff all vectors match within tolerance.
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
#include "bonsai.h"

extern "C" {
void rpcmem_init(void);
void* rpcmem_alloc(int heapid, unsigned flags, int size);
void rpcmem_free(void* po);
int remote_session_control(unsigned req, void* data, unsigned len);
}
struct umod { int domain; int enable; };

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
    if (argc != 3) { fprintf(stderr, "usage: gate_npu model.npubin gate.bin\n"); return 2; }
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);

    Map m; if (!m.open(argv[1])) { fprintf(stderr, "open model failed\n"); return 2; }
    if (memcmp(m.p, "NPU1", 4) != 0) { fprintf(stderr, "bad magic\n"); return 2; }
    Reader r{m.p, m.n}; r.o = 4;
    uint32_t ver, cnt; r.takev(ver); r.takev(cnt);
    std::unordered_map<std::string, Tensor> tab;
    for (uint32_t i = 0; i < cnt; i++) {
        std::string name; Tensor t; uint16_t nl;
        r.takev(nl); name.assign((const char*)(m.p + r.o), nl); r.o += nl;
        r.takev(t.kind); r.takev(t.out); r.takev(t.in); r.takev(t.off); r.takev(t.len);
        tab[name] = t;
    }

    Map g; if (!g.open(argv[2])) { fprintf(stderr, "open gate failed\n"); return 2; }
    if (memcmp(g.p, "GATE1", 5) != 0) { fprintf(stderr, "bad gate magic\n"); return 2; }
    Reader gr{g.p, g.n}; gr.o = 5;
    uint32_t gver, gcnt; gr.takev(gver); gr.takev(gcnt);

    // Signature-free offload MUST be requested before any other FastRPC call,
    // including rpcmem_init (Hexagon SDK rpc.html).
    struct umod um{3, 1};
    int uctrl = remote_session_control(2, &um, sizeof(um));
    printf("UNSIGNED_CTRL rc=%d\n", uctrl);
    rpcmem_init();
    remote_handle64 h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    if (rc != 0) { printf("NPU OPEN FAIL rc=%d\n", rc); return 1; }
    printf("NPU OPEN OK handle=%llu\n", (unsigned long long)h);

    double max_err = 0; int fails = 0; int skipped = 0;
    for (uint32_t i = 0; i < gcnt; i++) {
        std::string name; gr.str(name);
        uint32_t row, ind; gr.takev(row); gr.takev(ind);
        const float* x = (const float*)(g.p + gr.o); gr.o += (size_t)ind * 4;
        double yref; gr.takev(yref);
        auto it = tab.find(name);
        if (it == tab.end()) { printf("MISS %s\n", name.c_str()); fails++; continue; }
        const Tensor& t = it->second;
        if (t.kind != 0 || t.in != ind || row >= t.out) { fails++; continue; }
        uint32_t ng = ind / 128;
        uint32_t row_bytes = ind / 8;
        uint32_t prow = row_bytes + ((128 - (row_bytes % 128)) % 128);
        const uint8_t* sbase = m.p + t.off;
        const uint8_t* bbase = m.p + t.off + (size_t)t.out * ng * 2;
        size_t xbytes = (size_t)ind * 4;
        size_t bbytes = prow, sbytes = (size_t)ng * 2;
        float* dx = (float*)rpcmem_alloc(25, 1, (int)xbytes);
        unsigned char* db = (unsigned char*)rpcmem_alloc(25, 1, (int)bbytes);
        short* ds = (short*)rpcmem_alloc(25, 1, (int)sbytes);
        float* dy = (float*)rpcmem_alloc(25, 1, 4);
        if (!dx || !db || !ds || !dy) { printf("RPCMEM FAIL\n"); return 1; }
        memcpy(dx, x, xbytes);
        memcpy(db, bbase + (size_t)row * prow, bbytes);
        memcpy(ds, sbase + (size_t)row * ng * 2, sbytes);
        int ret = bonsai_gemv_q1(h, 1, (int)ind, (int)prow,
                                 dx, (int)ind, db, (int)row_bytes,
                                 ds, (int)ng, dy, 1);
        double y = dy[0];
        rpcmem_free(dx); rpcmem_free(db); rpcmem_free(ds); rpcmem_free(dy);
        if (ret != 0) { printf("INVOKE FAIL %s row %u rc=%d\n", name.c_str(), row, ret); fails++; continue; }
        double err = fabs(y - yref);
        if (err > max_err) max_err = err;
        if (err >= 0.05) { printf("MISMATCH %s row %u err %.6f ref %.6f got %.6f\n",
                                  name.c_str(), row, err, yref, y); fails++; }
    }
    printf("vectors %u fails %d skipped %d max_abs_err %.6f\n", gcnt, fails, skipped, max_err);
    printf(fails == 0 ? "NPU GATE PASS\n" : "NPU GATE FAIL\n");
    bonsai_close(h);
    return fails == 0 ? 0 : 1;
}
