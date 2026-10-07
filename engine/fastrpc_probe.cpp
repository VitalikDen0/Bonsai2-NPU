// fastrpc_probe: honest FastRPC plumbing check (open + rpcmem, NO fake compute).
// Opens the on-device CDSP stub URIs, reports return codes and latency.
// Exit 0 always (this is diagnostics, not a gate); output is the evidence.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <time.h>

typedef uint64_t remote_handle64;
struct remote_arg_t { void* buf; size_t len; };
struct umod { int domain; int enable; };

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main() {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    void* lib = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW);
    if (!lib) { printf("DLOPEN_FAIL %s\n", dlerror()); return 0; }
    printf("DLOPEN_OK\n");

    auto rpcmem_init = (void(*)())dlsym(lib, "rpcmem_init");
    auto rpcmem_alloc = (void*(*)(int, uint32_t, int))dlsym(lib, "rpcmem_alloc");
    auto rpcmem_free = (void(*)(void*))dlsym(lib, "rpcmem_free");
    auto sess = (int(*)(uint32_t, void*, uint32_t))dlsym(lib, "remote_session_control");
    auto open = (int(*)(const char*, remote_handle64*))dlsym(lib, "remote_handle64_open");
    auto close = (int(*)(remote_handle64))dlsym(lib, "remote_handle64_close");
    printf("SYMBOLS init=%d alloc=%d free=%d sess=%d open=%d close=%d\n",
           !!rpcmem_init, !!rpcmem_alloc, !!rpcmem_free, !!sess, !!open, !!close);
    if (!open) return 0;

    if (rpcmem_init) rpcmem_init();
    struct umod um{3, 1};
    if (sess) printf("UNSIGNED_CTRL rc=%d\n", sess(2, &um, sizeof(um)));

    // rpcmem DMA alloc timing (the Zero-Copy path we will use)
    if (rpcmem_alloc && rpcmem_free) {
        double t0 = now_ms();
        void* p = rpcmem_alloc(25, 1, 1 << 20);
        double t1 = now_ms();
        printf("RPCMEM_ALLOC_1MB ptr=%p latency_ms=%.3f\n", p, t1 - t0);
        if (p) { memset(p, 0xAB, 1 << 20); rpcmem_free(p); printf("RPCMEM_RW_FREE_OK\n"); }
    }

    const char* uris[] = {
        "file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp",
        "file:///libbonsai_htp_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_dom=cdsp",
        "file:///libbonsai_hvx_v79.so?bonsai_skel_handle_invoke&_modver=1.0&_dom=cdsp",
        "file:///libbonsai_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_dom=cdsp",
        "file:///libCalculator_skel.so?Calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    };
    for (auto uri : uris) {
        remote_handle64 h = (remote_handle64)-1;
        double t0 = now_ms();
        int rc = open(uri, &h);
        double t1 = now_ms();
        printf("OPEN rc=%d handle=%llu latency_ms=%.3f uri=%s\n",
               rc, (unsigned long long)h, t1 - t0, uri);
        if (rc == 0 && close) { printf("CLOSE rc=%d\n", close(h)); }
    }
    printf("PROBE_DONE\n");
    return 0;
}
