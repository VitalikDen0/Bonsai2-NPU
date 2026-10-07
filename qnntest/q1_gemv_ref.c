// q1_gemv_ref: host-side reference for binary g128 GEMV (matches npubin format).
// y[i] = sum_g s_g * sum_{j in group g}(bit(i,j) ? +x[j] : -x[j]), s_g = stored_scale/2.
// bits lsb-first, prow = bytes per row = K/8. Self-test vs naive double computation.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f = (e == 0) ? (s << 31) : (e == 31) ? ((s << 31) | 0x7f800000u | (m << 13))
                                                  : ((s << 31) | ((e + 112) << 23) | (m << 13));
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static uint16_t f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t s = (u >> 16) & 0x8000;
    int e = (int)((u >> 23) & 0xff) - 112;
    uint32_t m = u & 0x7fffff;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7bff);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

// kernel under test: same signature/semantics as hvx_gemv_q1 (float I/O)
int q1_gemv(int out_dim, int in_dim, int prow,
            const float* x, const unsigned char* bits,
            const short* scales, float* y) {
    if (!x || !bits || !scales || !y) return -1;
    if (out_dim <= 0 || in_dim <= 0 || (in_dim & 127) != 0 || prow <= 0) return -2;
    int ng = in_dim >> 7;
    for (int i = 0; i < out_dim; i++) {
        const uint16_t* sg = (const uint16_t*)scales + (unsigned)i * (unsigned)ng;
        const unsigned char* bp = bits + (unsigned)i * (unsigned)prow;
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            double s = (double)f16_to_f32(sg[g]) * 0.5;
            double gsum = 0.0;
            for (int j = 0; j < 128; j++) {
                int jj = g * 128 + j;
                int bit = (bp[jj >> 3] >> (jj & 7)) & 1;
                gsum += bit ? (double)x[jj] : -(double)x[jj];
            }
            acc += s * gsum;
        }
        y[i] = (float)acc;
    }
    return 0;
}

int main(void) {
    int fails = 0;
    // test 1: tiny exact case M=2 K=128, all bits=1, scale=2.0 -> s_g=1.0, y=sum(x)
    {
        float x[128], y[2];
        unsigned char bits[2 * 16];
        uint16_t sc[2];
        for (int j = 0; j < 128; j++) x[j] = (float)(j + 1) / 128.0f;
        memset(bits, 0xFF, sizeof(bits));
        sc[0] = f32_to_f16(2.0f); sc[1] = f32_to_f16(2.0f);
        int rc = q1_gemv(2, 128, 16, x, bits, (short*)sc, y);
        double expect = 0; for (int j = 0; j < 128; j++) expect += x[j];
        printf("t1 rc=%d y0=%.6f expect=%.6f y1=%.6f\n", rc, y[0], expect, y[1]);
        if (rc != 0 || fabs(y[0] - expect) > 1e-4 || fabs(y[1] - expect) > 1e-4) { printf("T1 FAIL\n"); fails++; }
    }
    // test 2: random vs naive double, M=8 K=512
    {
        int M = 8, K = 512, ng = K / 128, prow = K / 8;
        float* x = malloc(K * 4); float* y = malloc(M * 4); double* ref = malloc(M * 8);
        unsigned char* bits = malloc((size_t)M * prow);
        uint16_t* sc = malloc((size_t)M * ng * 2);
        unsigned seed = 123;
        for (int j = 0; j < K; j++) { seed = seed*1664525u+1013904223u; x[j] = ((seed>>8)/8388608.0f-1.0f)*0.1f; }
        for (int i = 0; i < M * prow; i++) { seed = seed*1664525u+1013904223u; bits[i] = (unsigned char)(seed >> 13); }
        for (int i = 0; i < M * ng; i++) { seed = seed*1664525u+1013904223u; sc[i] = f32_to_f16(0.5f + (seed>>8)/8388608.0f); }
        int rc = q1_gemv(M, K, prow, x, bits, (short*)sc, y);
        double maxerr = 0;
        for (int i = 0; i < M; i++) {
            double acc = 0;
            for (int g = 0; g < ng; g++) {
                double s = (double)f16_to_f32(sc[i*ng+g]) * 0.5;
                double gs = 0;
                for (int j = 0; j < 128; j++) { int jj = g*128+j; int b = (bits[i*prow+(jj>>3)] >> (jj&7)) & 1; gs += b ? x[jj] : -x[jj]; }
                acc += s * gs;
            }
            ref[i] = acc;
            double e = fabs(y[i] - ref[i]); if (e > maxerr) maxerr = e;
        }
        printf("t2 rc=%d maxerr=%.3e\n", rc, maxerr);
        if (rc != 0 || maxerr > 1e-5) { printf("T2 FAIL\n"); fails++; }
        free(x); free(y); free(ref); free(bits); free(sc);
    }
    // test 3: arg validation
    {
        float x[128] = {0}, y[1]; unsigned char b[16] = {0}; short s[1] = {0};
        int r1 = q1_gemv(0, 128, 16, x, b, s, y);
        int r2 = q1_gemv(1, 100, 16, x, b, s, y);
        int r3 = q1_gemv(1, 128, 16, NULL, b, s, y);
        printf("t3 r1=%d r2=%d r3=%d\n", r1, r2, r3);
        if (r1 != -2 || r2 != -2 || r3 != -1) { printf("T3 FAIL\n"); fails++; }
    }
    printf(fails ? "SELFTEST FAIL\n" : "SELFTEST PASS\n");
    return fails;
}
