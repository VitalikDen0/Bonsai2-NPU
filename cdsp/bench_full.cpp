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

    int out_dim = 17408;
    int in_dim = 5120;
    int ng = in_dim / 128;
    int prow = in_dim / 8; // 640 bytes

    size_t bits_sz = (size_t)out_dim * prow;
    size_t scales_sz = (size_t)out_dim * ng * 2;

    float* x = (float*)rpcmem_alloc(25, 1, in_dim * 4);
    unsigned char* bits = (unsigned char*)rpcmem_alloc(25, 1, bits_sz);
    short* scales = (short*)rpcmem_alloc(25, 1, scales_sz);
    float* y = (float*)rpcmem_alloc(25, 1, out_dim * 4);

    if (!x || !bits || !scales || !y) {
        printf("RPCMEM ALLOC FAILED!\n");
        return 1;
    }

    for (int i = 0; i < in_dim; i++) x[i] = 0.01f * (i % 7);
    memset(bits, 0xAA, bits_sz);
    // FP16 1.0 = 0x3C00
    for (size_t i = 0; i < (size_t)out_dim * ng; i++) scales[i] = (short)0x3C00;

    printf("Starting 17408x5120 GEMV benchmark on Hexagon NPU...\n");
    // Warmup
    bonsai_gemv_q1(h, out_dim, in_dim, prow, x, in_dim, bits, bits_sz, scales, out_dim * ng, y, out_dim);

    const int iters = 5;
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        int r = bonsai_gemv_q1(h, out_dim, in_dim, prow, x, in_dim, bits, bits_sz, scales, out_dim * ng, y, out_dim);
        if (r) { printf("INVOKE FAIL rc=%d\n", r); return 1; }
    }
    double t1 = now_ms();
    double avg_ms = (t1 - t0) / iters;
    double total_bytes = (double)(bits_sz + scales_sz + in_dim * 4 + out_dim * 4);
    printf(">>> 17408x5120 GEMV: %.2f ms, Bandwidth: %.2f GB/s, y[0]=%.4f, y[100]=%.4f <<<\n",
           avg_ms, (total_bytes / 1e9) / (avg_ms / 1000.0), y[0], y[100]);

    rpcmem_free(x); rpcmem_free(bits); rpcmem_free(scales); rpcmem_free(y);
    bonsai_close(h);
    return 0;
}
