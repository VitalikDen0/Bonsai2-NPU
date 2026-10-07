#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define HVX_MAX_K 17408
#define HVX_MAX_NG (HVX_MAX_K >> 7)
static float hvx_xalign[HVX_MAX_K] __attribute__((aligned(128)));
static int8_t hvx_qx8[HVX_MAX_K] __attribute__((aligned(128)));
static float hvx_sx[HVX_MAX_NG] __attribute__((aligned(128)));
static float hvx_sx4[HVX_MAX_NG * 4] __attribute__((aligned(128)));

static __attribute__((noinline)) void quantize_x_q8_permuted(const float* x, int batch, int in_dim) {
    int ng = in_dim >> 7;
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        const float* xb = x + (unsigned)b_idx * (unsigned)in_dim;
        int8_t* qdst_b = hvx_qx8 + (unsigned)b_idx * (unsigned)in_dim;
        float* sdst_b = hvx_sx + (unsigned)b_idx * (unsigned)ng;
        float* sdst4_b = hvx_sx4 + (unsigned)b_idx * (unsigned)(ng << 2);
        for (int g8 = 0; g8 < ng; g8 += 8) {
            int8_t* qdst_super = qdst_b + (unsigned)g8 * 128u;
            for (int sub = 0; sub < 8; sub++) {
                int g = g8 + sub;
                const float* gx = xb + (unsigned)g * 128u;
                float amax = 0.0f;
                for (int j = 0; j < 128; j++) {
                    float a = gx[j] < 0.0f ? -gx[j] : gx[j];
                    if (a > amax) amax = a;
                }
                float sx = amax * (1.0f / 127.0f);
                float inv_sx = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                sdst_b[g] = sx;
                sdst4_b[(g << 2) + 0] = sx;
                sdst4_b[(g << 2) + 1] = sx;
                sdst4_b[(g << 2) + 2] = sx;
                sdst4_b[(g << 2) + 3] = sx;
                for (int k = 0; k < 8; k++) {
                    int8_t* qdst_k = qdst_super + (unsigned)k * 128u + (unsigned)sub * 16u;
                    for (int m = 0; m < 4; m++) {
                        for (int b = 0; b < 4; b++) {
                            int j = (m << 5) + (b << 3) + k;
                            float v = gx[j] * inv_sx;
                            int q = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
                            if (q > 127) q = 127;
                            if (q < -127) q = -127;
                            qdst_k[(m << 2) + b] = (int8_t)q;
                        }
                    }
                }
            }
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    setbuf(stdout, NULL);
    for (int i = 0; i < HVX_MAX_K; i++) hvx_xalign[i] = (float)i * 0.001f;

    // Simulate 1 token's worth of calls:
    // 64 layers * (2 calls with dim 5120, 1 call with dim 6144, 1 call with dim 17408)
    int iters = 10;
    double t0 = now_ms();
    volatile float sum = 0.0f;
    for (int it = 0; it < iters; it++) {
        for (int L = 0; L < 64; L++) {
            quantize_x_q8_permuted(hvx_xalign, 1, 5120);
            sum += hvx_qx8[0] + hvx_sx[0];
            quantize_x_q8_permuted(hvx_xalign, 1, 6144);
            sum += hvx_qx8[0] + hvx_sx[0];
            quantize_x_q8_permuted(hvx_xalign, 1, 5120);
            sum += hvx_qx8[0] + hvx_sx[0];
            quantize_x_q8_permuted(hvx_xalign, 1, 17408);
            sum += hvx_qx8[0] + hvx_sx[0];
        }
    }
    double total_ms = (now_ms() - t0) / iters;
    printf("[probe_quant] Time per token for quantize_x_q8_permuted on ARM: %.3f ms (sum=%.1f)\n", total_ms, sum);
    return 0;
}
