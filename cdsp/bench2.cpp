// bench2: malloc-buffers vs rpcmem-buffers invoke cost (same payload).
// Usage: bench2
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

    // gate_proj-sized payload: out=17408 rows is heavy; use out=2048, in=5120
    const int out_dim = 2048, in_dim = 5120, ng = in_dim / 128, prow = in_dim / 8;
    const int iters = 10;
    double bytes = (double)((size_t)out_dim * prow + (size_t)out_dim * ng * 2 + in_dim * 4);

    // 1) malloc buffers (copied path)
    {
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
        printf("MALLOC ms_per_invoke=%.2f MBps=%.1f y0=%.4f\n",
               (t1 - t0) / iters, bytes * iters / (t1 - t0) * 1000.0 / 1e6, y[0]);
        free(x); free(bits); free(scales); free(y);
    }
    // 2) rpcmem buffers (possible zero-copy path)
    {
        float* x = (float*)rpcmem_alloc(25, 1, in_dim * 4);
        unsigned char* bits = (unsigned char*)rpcmem_alloc(25, 1, out_dim * prow);
        short* scales = (short*)rpcmem_alloc(25, 1, out_dim * ng * 2);
        float* y = (float*)rpcmem_alloc(25, 1, out_dim * 4);
        if (!x || !bits || !scales || !y) { printf("RPCMEM ALLOC FAIL\n"); return 1; }
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
        printf("RPCMEM ms_per_invoke=%.2f MBps=%.1f y0=%.4f\n",
               (t1 - t0) / iters, bytes * iters / (t1 - t0) * 1000.0 / 1e6, y[0]);
        rpcmem_free(x); rpcmem_free(bits); rpcmem_free(scales); rpcmem_free(y);
    }
    bonsai_close(h);
    printf("BENCH2 DONE\n");
    return 0;
}
