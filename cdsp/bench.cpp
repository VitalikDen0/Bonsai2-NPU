// bench: FastRPC invoke cost vs buffer size + max rpcmem alloc probe.
// Usage: bench
// Prints per-size median ms over iters + max single rpcmem alloc that succeeds.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {
void rpcmem_init(void);
void* rpcmem_alloc(int heapid, unsigned flags, int size);
void rpcmem_free(void* po);
}
#include "bonsai.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    extern int remote_session_control(unsigned, void*, unsigned);
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();
    remote_handle64 h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    if (rc) { printf("OPEN FAIL rc=%d\n", rc); return 1; }

    // max rpcmem alloc probe, MB-granular (int-safe sizes)
    size_t best = 0;
    for (int mb = 64; mb <= 8192; mb += 64) {
        void* p = rpcmem_alloc(25, 1, mb << 20);
        if (!p) break;
        rpcmem_free(p);
        best = mb;
    }
    printf("RPCMEM_MAX_MB=%d\n", (int)best);

    // invoke cost vs payload: out=256 rows fixed math, vary in_dim
    const int iters = 10;
    int dims[] = {512, 2048, 5120, 17408};
    for (int d = 0; d < 4; d++) {
        int in_dim = dims[d], out_dim = 64, ng = in_dim / 128, prow = in_dim / 8;
        float* x = (float*)malloc((size_t)in_dim * 4);
        unsigned char* bits = (unsigned char*)malloc((size_t)out_dim * prow);
        short* scales = (short*)malloc((size_t)out_dim * ng * 2);
        float* y = (float*)malloc((size_t)out_dim * 4);
        for (int i = 0; i < in_dim; i++) x[i] = 0.01f * (i % 7);
        memset(bits, 0xAA, (size_t)out_dim * prow);
        memset(scales, 0x3C, (size_t)out_dim * ng * 2);
        double t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            int r = bonsai_gemv_q1(h, out_dim, in_dim, prow, x, in_dim,
                                   bits, out_dim * prow, scales, out_dim * ng, y, out_dim);
            if (r) { printf("INVOKE FAIL rc=%d\n", r); return 1; }
        }
        double t1 = now_ms();
        double bytes = (double)((size_t)out_dim * prow + (size_t)out_dim * ng * 2 + in_dim * 4) * iters;
        printf("BENCH in=%d ms_per_invoke=%.2f MB_per_invoke=%.2f MBps=%.1f y0=%.4f\n",
               in_dim, (t1 - t0) / iters, bytes / iters / 1e6, bytes / (t1 - t0) * 1000.0 / 1e6, y[0]);
        free(x); free(bits); free(scales); free(y);
    }
    bonsai_close(h);
    printf("BENCH DONE\n");
    return 0;
}
