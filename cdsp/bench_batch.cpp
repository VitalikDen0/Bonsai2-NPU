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

    int max_batch = 32;
    float* x_all = (float*)rpcmem_alloc(25, 1, max_batch * in_dim * 4);
    unsigned char* bits = (unsigned char*)rpcmem_alloc(25, 1, bits_sz);
    short* scales = (short*)rpcmem_alloc(25, 1, scales_sz);
    float* y_all = (float*)rpcmem_alloc(25, 1, max_batch * out_dim * 4);

    if (!x_all || !bits || !scales || !y_all) {
        printf("ALLOC FAILED\n");
        return 1;
    }

    for (int i = 0; i < max_batch * in_dim; i++) x_all[i] = 0.01f * (i % 7);
    memset(bits, 0xAA, bits_sz);
    for (size_t i = 0; i < (size_t)out_dim * ng; i++) scales[i] = (short)0x3C00;

    printf("===================================================================================\n");
    printf("   BONSAI 2 NPU TRUE BATCH GEMM BENCHMARK (17408x5120 on Hexagon v79 6-Thread)\n");
    printf("===================================================================================\n");

    float* y_ref = (float*)malloc(max_batch * out_dim * 4);
    for (int b = 0; b < max_batch; b++) {
        for (int i = 0; i < in_dim; i++) {
            x_all[b * in_dim + i] = 0.01f * ((i + b) % 7);
        }
    }

    int batches[] = {1, 2, 4, 8, 16, 32};
    for (int b_idx = 0; b_idx < 6; b_idx++) {
        int B = batches[b_idx];
        const int iters = 10;

        // Warmup (single batched NPU GEMM call for all B tokens!)
        bonsai_gemv_q1(h, out_dim, in_dim, prow,
                       x_all, B * in_dim,
                       bits, bits_sz, scales, out_dim * ng,
                       y_all, B * out_dim);

        if (B == 1) {
            memcpy(y_ref, y_all, out_dim * 4);
        }

        double t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            bonsai_gemv_q1(h, out_dim, in_dim, prow,
                           x_all, B * in_dim,
                           bits, bits_sz, scales, out_dim * ng,
                           y_all, B * out_dim);
        }
        double t1 = now_ms();
        double total_ms = (t1 - t0) / iters;
        double ms_per_token = total_ms / B;
        double gflops = (2.0 * B * out_dim * in_dim) / 1e9;
        double tflops = gflops / (total_ms / 1000.0);

        float max_diff = 0.0f;
        for (int i = 0; i < out_dim; i++) {
            float d = y_all[i] - y_ref[i];
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
        }

        printf("Batch B=%2d: total=%6.2f ms | per-token=%5.2f ms | Speedup=%4.2fx | Throughput=%6.1f tok/s | Compute=%7.2f GFLOPS | diff=%.6f\n",
               B, total_ms, ms_per_token, 5.86 / ms_per_token, 1000.0 / ms_per_token, tflops * 1000.0, max_diff);
    }

    rpcmem_free(x_all); rpcmem_free(bits); rpcmem_free(scales); rpcmem_free(y_all);
    bonsai_close(h);
    return 0;
}
