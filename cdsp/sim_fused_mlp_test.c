#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline float fast_expf(float x) {
    x = x < -87.0f ? -87.0f : (x > 87.0f ? 87.0f : x);
    float z = x * 1.4426950408889634f;
    int ki = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));
    float kf = (float)ki;
    float r = (x - kf * 0.693145751953125f) - kf * 1.428606765330187045e-6f;
    float p = 1.98412698e-4f;
    p = p * r + 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 5.00000000e-1f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)(ki + 127) << 23;
    return p * u.f;
}

static inline float silu_f(float v) { return v / (1.0f + fast_expf(-v)); }

void bonsai_swiglu(const float* gate, const float* up, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = silu_f(gate[i]) * up[i];
}

#define FWHT_RADIX8(a0, a1, a2, a3, a4, a5, a6, a7) do { \
    float u0 = (a0) + (a1), u1 = (a0) - (a1); \
    float u2 = (a2) + (a3), u3 = (a2) - (a3); \
    float u4 = (a4) + (a5), u5 = (a4) - (a5); \
    float u6 = (a6) + (a7), u7 = (a6) - (a7); \
    float v0 = u0 + u2, v1 = u1 + u3, v2 = u0 - u2, v3 = u1 - u3; \
    float v4 = u4 + u6, v5 = u5 + u7, v6 = u4 - u6, v7 = u5 - u7; \
    (a0) = v0 + v4; (a1) = v1 + v5; (a2) = v2 + v6; (a3) = v3 + v7; \
    (a4) = v0 - v4; (a5) = v1 - v5; (a6) = v2 - v6; (a7) = v3 - v7; \
} while (0)

void bonsai_fwht1024(const float* x, const float* signs, int n, int inverse, float* out) {
    const float scale = 0.03125f;
    float b[1024];
    for (int blk = 0; blk < n; blk += 1024) {
        const float* xb = x + blk;
        const float* sb = signs ? (signs + blk) : NULL;
        float* dst = out + blk;

        // Pass 1: Fused input sign multiply + stages h = 1, 2, 4 (Radix-8) into L1-resident b[1024]
        if (!inverse && sb) {
            for (int i = 0; i < 1024; i += 8) {
                float a0 = xb[i+0] * sb[i+0], a1 = xb[i+1] * sb[i+1];
                float a2 = xb[i+2] * sb[i+2], a3 = xb[i+3] * sb[i+3];
                float a4 = xb[i+4] * sb[i+4], a5 = xb[i+5] * sb[i+5];
                float a6 = xb[i+6] * sb[i+6], a7 = xb[i+7] * sb[i+7];
                FWHT_RADIX8(a0, a1, a2, a3, a4, a5, a6, a7);
                b[i+0] = a0; b[i+1] = a1; b[i+2] = a2; b[i+3] = a3;
                b[i+4] = a4; b[i+5] = a5; b[i+6] = a6; b[i+7] = a7;
            }
        } else {
            for (int i = 0; i < 1024; i += 8) {
                float a0 = xb[i+0], a1 = xb[i+1], a2 = xb[i+2], a3 = xb[i+3];
                float a4 = xb[i+4], a5 = xb[i+5], a6 = xb[i+6], a7 = xb[i+7];
                FWHT_RADIX8(a0, a1, a2, a3, a4, a5, a6, a7);
                b[i+0] = a0; b[i+1] = a1; b[i+2] = a2; b[i+3] = a3;
                b[i+4] = a4; b[i+5] = a5; b[i+6] = a6; b[i+7] = a7;
            }
        }

        // Pass 2: Fused stages h = 8, 16, 32 (Radix-8 with stride 8) in L1-resident b[1024]
        for (int i = 0; i < 1024; i += 64) {
            for (int j = i; j < i + 8; j++) {
                float a0 = b[j +  0], a1 = b[j +  8], a2 = b[j + 16], a3 = b[j + 24];
                float a4 = b[j + 32], a5 = b[j + 40], a6 = b[j + 48], a7 = b[j + 56];
                FWHT_RADIX8(a0, a1, a2, a3, a4, a5, a6, a7);
                b[j +  0] = a0; b[j +  8] = a1; b[j + 16] = a2; b[j + 24] = a3;
                b[j + 32] = a4; b[j + 40] = a5; b[j + 48] = a6; b[j + 56] = a7;
            }
        }

        // Pass 3: Fused stages h = 64, 128, 256, 512 (Radix-16) + final scale store directly to dst
        for (int j = 0; j < 64; j++) {
            float t[16];
            for (int s = 0; s < 16; s++) t[s] = b[j + s * 64];

            // In-register Radix-16: 4 sub-stages (64, 128, 256, 512)
            for (int step = 1; step < 16; step <<= 1) {
                for (int i = 0; i < 16; i += 2 * step) {
                    for (int k = 0; k < step; k++) {
                        float u = t[i + k];
                        float v = t[i + k + step];
                        t[i + k] = u + v;
                        t[i + k + step] = u - v;
                    }
                }
            }
            for (int s = 0; s < 16; s++) dst[j + s * 64] = t[s] * scale;
        }
    }
}

static inline unsigned long long read_cycles(void) {
    unsigned long long c;
    asm volatile ("%0 = c15:14" : "=r"(c));
    return c;
}

int main(void) {
    static float g[17408], u[17408], o[17408], signs[17408], out[17408];
    for (int i = 0; i < 17408; i++) {
        g[i] = (float)(i % 100) * 0.05f - 2.5f;
        u[i] = (float)(i % 50) * 0.02f;
        signs[i] = (i & 1) ? 1.0f : -1.0f;
    }

    unsigned long long c0 = read_cycles();
    bonsai_swiglu(g, u, 17408, o);
    unsigned long long c1 = read_cycles();
    bonsai_fwht1024(o, signs, 17408, 0, out);
    unsigned long long c2 = read_cycles();

    printf("SwiGLU cycles: %llu (%.3f ms @ 2GHz)\n", c1 - c0, (double)(c1 - c0) / 2e6);
    printf("FWHT-1024 cycles: %llu (%.3f ms @ 2GHz)\n", c2 - c1, (double)(c2 - c1) / 2e6);
    printf("Sample out: %f %f %f\n", out[0], out[1024], out[17407]);
    return 0;
}
