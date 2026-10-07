// Q1MatMul core: binary GEMV y[i] = sum_g s[i*ng+g] * sum_j(sign(i,j)*x[j]).
// s = per-group fp32 scale (== stored_fp16/2, pre-divided at load).
// bits lsb-first, prow = K/8 bytes per row. Self-test vs double ref.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 0=ok, -1=null, -2=bad dims
static int q1matmul_validate(int M, int K, int prow, int ng, int nscales) {
    if (M <= 0 || K <= 0 || prow <= 0 || ng <= 0) return -1;
    if ((K & 127) != 0) return -2;
    if (prow != K / 8) return -2;
    if (ng != K / 128) return -2;
    if (nscales != M * ng) return -2;
    return 0;
}

static int q1matmul_execute(int M, int K, int prow,
                            const float* x, const unsigned char* bits,
                            const float* scales, float* y) {
    int ng = K >> 7;
    if (!x || !bits || !scales || !y) return -1;
    if (M <= 0 || K <= 0 || (K & 127) != 0 || prow != K / 8) return -2;
    for (int i = 0; i < M; i++) {
        const float* sg = scales + (unsigned)i * (unsigned)ng;
        const unsigned char* bp = bits + (unsigned)i * (unsigned)prow;
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            double s = (double)sg[g];
            double gsum = 0.0;
            int base = g * 128;
            for (int j = 0; j < 128; j++) {
                int jj = base + j;
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
    // t1: M=1 K=128, all +1, s=0.5 -> y=sum(x)*0.5... use s=1.0: y=sum(x)
    {
        float x[128], y[1], s[1] = {1.0f};
        unsigned char b[16];
        memset(b, 0xFF, sizeof(b));
        for (int j = 0; j < 128; j++) x[j] = (j + 1) * 0.01f;
        int rc = q1matmul_execute(1, 128, 16, x, b, s, y);
        double e = 0; for (int j = 0; j < 128; j++) e += x[j];
        printf("t1 rc=%d y=%.6f expect=%.6f\n", rc, y[0], e);
        if (rc != 0 || fabs(y[0] - e) > 1e-4) { printf("T1 FAIL\n"); fails++; }
    }
    // t2: random M=16 K=1024 vs double ref (independent loops)
    {
        int M = 16, K = 1024, ng = 8, prow = 128;
        float *x = (float*)malloc(K * 4), *y = (float*)malloc(M * 4), *s = (float*)malloc((size_t)M * ng * 4);
        unsigned char* b = (unsigned char*)malloc((size_t)M * prow);
        unsigned seed = 7;
        for (int j = 0; j < K; j++) { seed = seed*1664525u+1013904223u; x[j] = ((seed>>8)/8388608.0f-1.0f)*0.2f; }
        for (int i = 0; i < M * prow; i++) { seed = seed*1664525u+1013904223u; b[i] = (unsigned char)(seed >> 11); }
        for (int i = 0; i < M * ng; i++) { seed = seed*1664525u+1013904223u; s[i] = 0.005f + (seed>>8)/8388608.0f*0.03f; }
        int rc = q1matmul_execute(M, K, prow, x, b, s, y);
        int vrc = q1matmul_validate(M, K, prow, ng, M * ng);
        double maxerr = 0;
        for (int i = 0; i < M; i++) {
            long double acc = 0;
            for (int g = 0; g < ng; g++)
                for (int j = 0; j < 128; j++) {
                    int jj = g * 128 + j;
                    int bit = (b[i * prow + (jj >> 3)] >> (jj & 7)) & 1;
                    acc += (long double)s[i * ng + g] * (bit ? x[jj] : -x[jj]);
                }
            double e = fabs((double)acc - y[i]); if (e > maxerr) maxerr = e;
        }
        printf("t2 rc=%d vrc=%d maxerr=%.3e\n", rc, vrc, maxerr);
        if (rc != 0 || vrc != 0 || maxerr > 1e-5) { printf("T2 FAIL\n"); fails++; }
        free(x); free(y); free(s); free(b);
    }
    // t3: validate rejects bad dims
    {
        int a = q1matmul_validate(4, 100, 16, 1, 4);   // K%128
        int b = q1matmul_validate(4, 128, 20, 1, 4);   // prow
        int c = q1matmul_validate(4, 128, 16, 2, 4);   // ng
        int d = q1matmul_validate(4, 128, 16, 1, 5);   // nscales
        int e = q1matmul_validate(0, 128, 16, 1, 0);   // non-positive
        printf("t3 %d %d %d %d %d\n", a, b, c, d, e);
        if (a != -2 || b != -2 || c != -2 || d != -2 || e != -1) { printf("T3 FAIL\n"); fails++; }
    }
    // t4: K=5120 (real width) smoke, M=2
    {
        int M = 2, K = 5120, ng = 40, prow = 640;
        float *x = (float*)malloc(K * 4), *y = (float*)malloc(M * 4), *s = (float*)malloc((size_t)M * ng * 4);
        unsigned char* b = (unsigned char*)malloc((size_t)M * prow);
        unsigned seed = 99;
        for (int j = 0; j < K; j++) { seed = seed*1664525u+1013904223u; x[j] = ((seed>>8)/8388608.0f-1.0f)*0.05f; }
        for (int i = 0; i < M * prow; i++) { seed = seed*1664525u+1013904223u; b[i] = (unsigned char)(seed >> 13); }
        for (int i = 0; i < M * ng; i++) { seed = seed*1664525u+1013904223u; s[i] = 0.008f; }
        double t0 = 0; (void)t0;
        int rc = q1matmul_execute(M, K, prow, x, b, s, y);
        printf("t4 rc=%d y0=%.6f y1=%.6f finite=%d\n", rc, y[0], y[1], isfinite(y[0]) && isfinite(y[1]));
        if (rc != 0 || !isfinite(y[0]) || !isfinite(y[1])) { printf("T4 FAIL\n"); fails++; }
        free(x); free(y); free(s); free(b);
    }
    printf(fails ? "SELFTEST FAIL\n" : "SELFTEST PASS\n");
    return fails;
}
