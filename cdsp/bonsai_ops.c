// Bonsai transformer decode ops (fp32, CDSP-portable, no HVX yet).
// Matches Qwen3.5 config: rms_eps=1e-6, rope_theta=1e7, partial_rotary=0.25,
// mrope_section=[11,11,10], swiGLU (silu(gate)*up), temp 0.7 / top_p 0.95 / top_k 20.
#include <math.h>
#include <string.h>
#include <stdint.h>

void bonsai_rmsnorm(const float* x, const float* w, int n, float* out) {
    // Stored weights in both Bonsai 1 and Bonsai 2 already include the (1 + w) shift.
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(s / n) + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// partial RoPE, Qwen3.5 text path: first 64 dims (of 256) rotated in NeoX
// pairs (v[k], v[k+32]), k=0..31, freq_k = theta^(-k/32), theta=1e7.
// (mrope interleaving is identity for text-only single-token decode)
void bonsai_rope(float* q, float* k, int heads, int head_dim, int pos) {
    int rot = head_dim / 4;  // 64
    int half = rot / 2;       // 32
    static float s_freq[64];
    static int s_freq_init = 0;
    if (!s_freq_init) {
        for (int i = 0; i < 32; i++) {
            s_freq[i] = 1.0f / powf(1e7f, (2.0f * i) / 64.0f);
        }
        s_freq_init = 1;
    }
    float cc[64], ss[64];
    for (int i = 0; i < half; i++) {
        float a = (float)pos * s_freq[i];
        cc[i] = cosf(a);
        ss[i] = sinf(a);
    }
    for (int h = 0; h < heads; h++) {
        float* qq = q + h * head_dim;
        float* kk = k ? k + h * head_dim : 0;
        for (int i = 0; i < half; i++) {
            float c = cc[i], s = ss[i];
            float q0 = qq[i], q1 = qq[i + half];
            qq[i] = q0 * c - q1 * s;
            qq[i + half] = q0 * s + q1 * c;
            if (kk) {
                float k0 = kk[i], k1 = kk[i + half];
                kk[i] = k0 * c - k1 * s;
                kk[i + half] = k0 * s + k1 * c;
            }
        }
    }
}

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
    p = p * r * r + r + 1.0f;
    uint32_t bits = (uint32_t)((ki + 127) << 23);
    float scale;
    memcpy(&scale, &bits, 4);
    return p * scale;
}

static inline float silu_f(float v) { return v / (1.0f + fast_expf(-v)); }

void bonsai_swiglu(const float* gate, const float* up, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = silu_f(gate[i]) * up[i];
}

// softmax over n (n<=262144), out = probs. Returns max logit.
float bonsai_softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    float inv = (float)(1.0 / s);
    for (int i = 0; i < n; i++) x[i] *= inv;
    return mx;
}

// greedy-or-sample next token id from logits over vocab with temp/top_k/top_p.
// deterministic LCG; seed updated in place. Returns token id.
int bonsai_sample(const float* logits, int vocab, float temp,
                  int top_k, float top_p, unsigned* seed) {
    // temp<=0 -> greedy
    int bi = 0;
    if (temp <= 0) {
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[bi]) bi = i;
        return bi;
    }
    // partial select top_k (simple insertion over small k<=64)
    static int idx[256];
    static float vv[256];
    int k = top_k < 256 ? top_k : 256;
    for (int i = 0; i < k; i++) { idx[i] = -1; vv[i] = -1e30f; }
    for (int i = 0; i < vocab; i++) {
        float v = logits[i] / temp;
        for (int j = 0; j < k; j++) {
            if (v > vv[j]) {
                for (int t = k - 1; t > j; t--) { vv[t] = vv[t - 1]; idx[t] = idx[t - 1]; }
                vv[j] = v; idx[j] = i;
                break;
            }
        }
    }
    // softmax over top_k, cumsum to top_p
    float mx = vv[0];
    double s = 0;
    for (int j = 0; j < k && idx[j] >= 0; j++) { vv[j] = expf(vv[j] - mx); s += vv[j]; }
    *seed = *seed * 1664525u + 1013904223u;
    double r = ((double)(*seed >> 8) / 16777216.0) * s;
    double acc = 0;
    int last = 0;
    for (int j = 0; j < k && idx[j] >= 0; j++) {
        acc += vv[j];
        last = idx[j];
        if (acc >= r || (j > 0 && acc / s >= top_p)) break;
    }
    return last;
}

// Fast Walsh-Hadamard Transform (normalized-sylvester-walsh-hadamard, block size 1024).
// 3-pass register-resident Radix-8 (h=1,2,4) + Radix-8 (h=8,16,32) + Radix-16 (h=64,128,256,512 + scale).
// Eliminates 9 of 12 memory passes and 100% of small-stride branch mispredictions.
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

        // Pass 3: Fused stages h = 64, 128, 256, 512 (Radix-16 = 2x Radix-8 + h=512 butterfly + scale) -> dst
        for (int j = 0; j < 64; j++) {
            float a0 = b[j +   0], a1 = b[j +  64], a2 = b[j + 128], a3 = b[j + 192];
            float a4 = b[j + 256], a5 = b[j + 320], a6 = b[j + 384], a7 = b[j + 448];
            float c0 = b[j + 512], c1 = b[j + 576], c2 = b[j + 640], c3 = b[j + 704];
            float c4 = b[j + 768], c5 = b[j + 832], c6 = b[j + 896], c7 = b[j + 960];
            FWHT_RADIX8(a0, a1, a2, a3, a4, a5, a6, a7);
            FWHT_RADIX8(c0, c1, c2, c3, c4, c5, c6, c7);
            if (inverse && sb) {
                dst[j +   0] = (a0 + c0) * scale * sb[j +   0]; dst[j + 512] = (a0 - c0) * scale * sb[j + 512];
                dst[j +  64] = (a1 + c1) * scale * sb[j +  64]; dst[j + 576] = (a1 - c1) * scale * sb[j + 576];
                dst[j + 128] = (a2 + c2) * scale * sb[j + 128]; dst[j + 640] = (a2 - c2) * scale * sb[j + 640];
                dst[j + 192] = (a3 + c3) * scale * sb[j + 192]; dst[j + 704] = (a3 - c3) * scale * sb[j + 704];
                dst[j + 256] = (a4 + c4) * scale * sb[j + 256]; dst[j + 768] = (a4 - c4) * scale * sb[j + 768];
                dst[j + 320] = (a5 + c5) * scale * sb[j + 320]; dst[j + 832] = (a5 - c5) * scale * sb[j + 832];
                dst[j + 384] = (a6 + c6) * scale * sb[j + 384]; dst[j + 896] = (a6 - c6) * scale * sb[j + 896];
                dst[j + 448] = (a7 + c7) * scale * sb[j + 448]; dst[j + 960] = (a7 - c7) * scale * sb[j + 960];
            } else {
                dst[j +   0] = (a0 + c0) * scale; dst[j + 512] = (a0 - c0) * scale;
                dst[j +  64] = (a1 + c1) * scale; dst[j + 576] = (a1 - c1) * scale;
                dst[j + 128] = (a2 + c2) * scale; dst[j + 640] = (a2 - c2) * scale;
                dst[j + 192] = (a3 + c3) * scale; dst[j + 704] = (a3 - c3) * scale;
                dst[j + 256] = (a4 + c4) * scale; dst[j + 768] = (a4 - c4) * scale;
                dst[j + 320] = (a5 + c5) * scale; dst[j + 832] = (a5 - c5) * scale;
                dst[j + 384] = (a6 + c6) * scale; dst[j + 896] = (a6 - c6) * scale;
                dst[j + 448] = (a7 + c7) * scale; dst[j + 960] = (a7 - c7) * scale;
            }
        }
    }
}


