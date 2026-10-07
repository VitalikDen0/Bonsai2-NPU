#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static inline float fast_expf_m(float x) {
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
    p = p * r * r + r + 1.0f;
    uint32_t bits = (uint32_t)((ki + 127) << 23);
    float scale;
    memcpy(&scale, &bits, 4);
    return p * scale;
}

static inline float sigf(float x) { return 1.0f / (1.0f + fast_expf_m(-x)); }

// Verify DeltaNet recurrence math
static float S[48 * 128 * 128];
static float S_ref[48 * 128 * 128];

int main(void) {
    printf("[sim] Verifying DeltaNet recurrence and Linear Attention math...\n");
    memset(S, 0, sizeof(S));
    memset(S_ref, 0, sizeof(S_ref));

    float q[2048], k[2048], v[6144], z[6144], a[48], b[48];
    float nw[128], alog[48], dtb[48];

    for (int i = 0; i < 2048; i++) { q[i] = sinf((float)i * 0.01f); k[i] = cosf((float)i * 0.01f); }
    for (int i = 0; i < 6144; i++) { v[i] = sinf((float)i * 0.02f); z[i] = cosf((float)i * 0.02f); }
    for (int i = 0; i < 48; i++) { a[i] = 0.1f * (float)(i % 5); b[i] = -0.2f * (float)(i % 3); alog[i] = -0.5f; dtb[i] = 0.05f; }
    for (int i = 0; i < 128; i++) nw[i] = 1.0f;

    float no[6144], no_ref[6144];

    // Reference implementation
    for (int h = 0; h < 48; h++) {
        float oh[128];
        float* Sh = S_ref + (size_t)h * 128 * 128;
        float al = -expf(alog[h]);
        float db = dtb[h];
        float beta = sigf(b[h]);
        float gv = al * log1pf(expf(a[h] + db));
        float eg = expf(gv);
        const float* qh = q + (size_t)(h / 3) * 128;
        const float* kh = k + (size_t)(h / 3) * 128;
        const float* vh = v + (size_t)h * 128;

        float kq = 0.0f;
        for (int i = 0; i < 128; i++) kq += kh[i] * qh[i];
        double norm_sq = 0.0;

        for (int j = 0; j < 128; j++) {
            float* Sj = Sh + (size_t)j * 128;
            float sk = 0.0f, sq = 0.0f;
            for (int i = 0; i < 128; i++) {
                sk += Sj[i] * kh[i];
                sq += Sj[i] * qh[i];
            }
            float kv = sk * eg;
            float delta = (vh[j] - kv) * beta;
            float oj = sq * eg + delta * kq;
            oh[j] = oj;
            norm_sq += (double)oj * oj;

            for (int i = 0; i < 128; i++) {
                Sj[i] = Sj[i] * eg + kh[i] * delta;
            }
        }
        const float* zr = z + (size_t)h * 128;
        float* noh = no_ref + (size_t)h * 128;
        float inv = 1.0f / sqrtf((float)(norm_sq / 128.0) + 1e-6f);
        for (int j = 0; j < 128; j++) {
            noh[j] = oh[j] * inv * nw[j] * (zr[j] / (1.0f + fast_expf_m(-zr[j])));
        }
    }

    // Vector unrolled implementation (test candidate for CDSP)
    for (int h = 0; h < 48; h++) {
        float oh[128];
        float* Sh = S + (size_t)h * 128 * 128;
        float al = -expf(alog[h]);
        float db = dtb[h];
        float beta = sigf(b[h]);
        float gv = al * log1pf(expf(a[h] + db));
        float eg = expf(gv);
        const float* qh = q + (size_t)(h / 3) * 128;
        const float* kh = k + (size_t)(h / 3) * 128;
        const float* vh = v + (size_t)h * 128;

        float kq0 = 0.0f, kq1 = 0.0f, kq2 = 0.0f, kq3 = 0.0f;
        for (int i = 0; i < 128; i += 4) {
            kq0 += kh[i+0] * qh[i+0]; kq1 += kh[i+1] * qh[i+1];
            kq2 += kh[i+2] * qh[i+2]; kq3 += kh[i+3] * qh[i+3];
        }
        float kq = (kq0 + kq1) + (kq2 + kq3);
        double norm_sq = 0.0;

        for (int j = 0; j < 128; j++) {
            float* Sj = Sh + (size_t)j * 128;
            float sk0 = 0.0f, sk1 = 0.0f, sk2 = 0.0f, sk3 = 0.0f;
            float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
            for (int i = 0; i < 128; i += 4) {
                sk0 += Sj[i+0] * kh[i+0]; sk1 += Sj[i+1] * kh[i+1];
                sk2 += Sj[i+2] * kh[i+2]; sk3 += Sj[i+3] * kh[i+3];
                sq0 += Sj[i+0] * qh[i+0]; sq1 += Sj[i+1] * qh[i+1];
                sq2 += Sj[i+2] * qh[i+2]; sq3 += Sj[i+3] * qh[i+3];
            }
            float kv = ((sk0 + sk1) + (sk2 + sk3)) * eg;
            float delta = (vh[j] - kv) * beta;
            float oj = ((sq0 + sq1) + (sq2 + sq3)) * eg + delta * kq;
            oh[j] = oj;
            norm_sq += (double)oj * oj;

            for (int i = 0; i < 128; i += 4) {
                Sj[i+0] = Sj[i+0] * eg + kh[i+0] * delta;
                Sj[i+1] = Sj[i+1] * eg + kh[i+1] * delta;
                Sj[i+2] = Sj[i+2] * eg + kh[i+2] * delta;
                Sj[i+3] = Sj[i+3] * eg + kh[i+3] * delta;
            }
        }
        const float* zr = z + (size_t)h * 128;
        float* noh = no + (size_t)h * 128;
        float inv = 1.0f / sqrtf((float)(norm_sq / 128.0) + 1e-6f);
        for (int j = 0; j < 128; j++) {
            noh[j] = oh[j] * inv * nw[j] * (zr[j] / (1.0f + fast_expf_m(-zr[j])));
        }
    }

    // Verify difference
    float max_diff = 0.0f;
    for (int i = 0; i < 6144; i++) {
        float d = fabsf(no[i] - no_ref[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("[sim] DeltaNet Recurrence Validation: max_diff = %e (MATCH=%s)\n",
           max_diff, max_diff < 1e-5f ? "YES" : "NO");
    return max_diff < 1e-5f ? 0 : 1;
}
