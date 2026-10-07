// Bonsai Q1_0 GEMV implementation for CDSP.
// Correctness-first scalar C; HVX upgrade after gate passes on DSP.
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "bonsai.h"
#include "HAP_debug.h"
#include "HAP_power.h"

#pragma weak HAP_power_set
#pragma weak HAP_power_get
#pragma weak HAP_power_destroy_client
#pragma weak HAP_power_request
#pragma weak HVX_power_request

static void* g_pwr_ctx = NULL;

int bonsai_open(const char* uri, remote_handle64* handle) {
    (void)uri;
    if (HAP_power_request) {
        HAP_power_request(100, 100, 1);
    }
    if (HVX_power_request) {
        HVX_power_request();
    }
    if (HAP_power_set) {
        if (!g_pwr_ctx) g_pwr_ctx = malloc(1);
        HAP_power_request_t req;

        memset(&req, 0, sizeof(req));
        req.type = HAP_power_set_apptype;
        req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
        HAP_power_set(g_pwr_ctx, &req);

        memset(&req, 0, sizeof(req));
        req.type = HAP_power_set_HVX;
        req.hvx.power_up = TRUE;
        HAP_power_set(g_pwr_ctx, &req);

        memset(&req, 0, sizeof(req));
        req.type = HAP_power_set_DCVS_v3;
        req.dcvs_v3.set_dcvs_enable = TRUE;
        req.dcvs_v3.dcvs_enable = TRUE;
        req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        req.dcvs_v3.set_latency = TRUE;
        req.dcvs_v3.latency = 1;
        req.dcvs_v3.set_core_params = TRUE;
        req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
        req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
        req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
        req.dcvs_v3.set_bus_params = TRUE;
        req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
        req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
        req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
        req.dcvs_v3.set_sleep_disable = TRUE;
        req.dcvs_v3.sleep_disable = HAP_DCVS_LPM_LEVEL1;
        int rc_v3 = HAP_power_set(g_pwr_ctx, &req);

        if (rc_v3 != 0) {
            memset(&req, 0, sizeof(req));
            req.type = HAP_power_set_DCVS_v2;
            req.dcvs_v2.dcvs_enable = TRUE;
            req.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
            req.dcvs_v2.set_latency = TRUE;
            req.dcvs_v2.latency = 1;
            req.dcvs_v2.set_dcvs_params = TRUE;
            req.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_MAX;
            req.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_MAX;
            req.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_MAX;
            HAP_power_set(g_pwr_ctx, &req);
        }

        memset(&req, 0, sizeof(req));
        req.type = HAP_power_set_CENG_bus;
        req.ceng_bus.target_corner = HAP_DCVS_VCORNER_MAX;
        req.ceng_bus.min_corner = HAP_DCVS_VCORNER_MAX;
        req.ceng_bus.max_corner = HAP_DCVS_VCORNER_MAX;
        req.ceng_bus.perf_mode = HAP_CLK_PERF_HIGH;
        HAP_power_set(g_pwr_ctx, &req);

        memset(&req, 0, sizeof(req));
        req.type = HAP_power_set_mips_bw;
        req.mips_bw.set_mips = TRUE;
        req.mips_bw.mipsPerThread = 2000;
        req.mips_bw.mipsTotal = 12000;
        req.mips_bw.set_bus_bw = TRUE;
        req.mips_bw.bwBytePerSec = 50000000000ULL;
        req.mips_bw.busbwUsagePercentage = 100;
        req.mips_bw.set_latency = TRUE;
        req.mips_bw.latency = 1;
        HAP_power_set(g_pwr_ctx, &req);
    }
    *handle = (remote_handle64)0xB04A1;
    return 0;
}

int bonsai_close(remote_handle64 handle) {
    (void)handle;
    if (g_pwr_ctx) {
        HAP_power_destroy(g_pwr_ctx);
        free(g_pwr_ctx);
        g_pwr_ctx = NULL;
    }
    return 0;
}

static inline float f16_to_f32(unsigned short h) {
    unsigned s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    unsigned f;
    if (e == 0) f = s << 31;
    else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r;
    memcpy(&r, &f, 4);
    return r;
}

int hvx_gemv_q1(int out_dim, int in_dim, int prow,
                const float* x, const unsigned char* bits,
                const short* scales, float* y);

int hvx_gemm_q1(int batch, int out_dim, int in_dim, int prow,
                const float* x, const unsigned char* bits,
                const short* scales, float* y);

int hvx_swiglu_fwht1024(int batch, const float* gate_up, const float* signs, float* out);

static unsigned long long join_ptr(int hi, int lo) {
    return (((unsigned long long)(unsigned)hi) << 32) | (unsigned)lo;
}

static int rseek_to(unsigned long long target);
static int rread_full(unsigned char* dst, size_t n);

int bonsai_fprobe(remote_handle64 h, int off_hi, int off_lo,
                  unsigned char* data, int dataLen) {
    (void)h;
    if (!data || dataLen < 64) return -1;
    unsigned long long off = (((unsigned long long)(unsigned)off_hi) << 32) | (unsigned)off_lo;
    // Shared persistent fd + SEEK strides (read-skip re-walks hung for GB ranges).
    int rc = rseek_to(off);
    if (rc) return 200 - rc;
    rc = rread_full(data, 64);
    if (rc) return 220 - rc;
    return 0;
}

// persistent file reader: absolute 64-bit offsets via SEEK_CUR strides
// (lseek long is 32-bit; strides stay within +-2GB per call).
static int g_rfd = -1;
static unsigned long long g_rpos = 0;

static int rseek_to(unsigned long long target) {
    if (g_rfd < 0) {
        g_rfd = open("/data/local/tmp/bonsai1bit/bonsai27b-1bit.npubin", 0);
        if (g_rfd < 0) return -10;
        g_rpos = 0;
    }
    while (g_rpos != target) {
        if (g_rpos < target) {
            unsigned long long step = target - g_rpos;
            long chunk = step > 0x40000000u ? 0x40000000 : (long)step;
            // NOTE: positions >=2GB read back negative as signed long;
            // only (long)-1 counts as error (our stride landings and blob
            // offsets can never equal exactly 0xFFFFFFFF).
            if (lseek(g_rfd, chunk, 1 /*SEEK_CUR*/) == -1) return -11;
            g_rpos += (unsigned long long)chunk;
        } else {
            unsigned long long step = g_rpos - target;
            long chunk = step > 0x40000000u ? 0x40000000 : (long)step;
            if (lseek(g_rfd, -chunk, 1 /*SEEK_CUR*/) == -1) return -12;
            g_rpos -= (unsigned long long)chunk;
        }
    }
    return 0;
}

static int rread_full(unsigned char* dst, size_t n) {
    while (n > 0) {
        ssize_t r = read(g_rfd, dst, n > 0x40000000u ? 0x40000000u : n);
        if (r <= 0) return -13;
        dst += r;
        n -= (size_t)r;
        g_rpos += (unsigned long long)r;
    }
    return 0;
}

static unsigned char bread_scratch[1 << 16];  // TEMP chunk-size experiment

int bonsai_bread(remote_handle64 h, int off_hi, int off_lo, int len, long long* csum) {
    (void)h;
    if (!csum || len <= 0 || len > (1 << 28)) return -1;
    unsigned long long off = (((unsigned long long)(unsigned)off_hi) << 32) | (unsigned)off_lo;
    int rc = rseek_to(off);
    if (rc) return 200 - rc;
    unsigned long long hsh = 1469598103934665603ull;
    int left = len;
    while (left > 0) {
        int chunk = left > (int)sizeof(bread_scratch) ? (int)sizeof(bread_scratch) : left;
        rc = rread_full(bread_scratch, (size_t)chunk);
        if (rc) return 300 - rc;
        for (int i = 0; i < chunk; i++) {
            hsh ^= bread_scratch[i];
            hsh *= 1099511628211ull;
        }
        left -= chunk;
    }
    *csum = (long long)hsh;
    return 0;
}

int bonsai_fmaptest(remote_handle64 h, int win_hi, int win_lo, int at_off,
                    unsigned char* data, int dataLen) {
    (void)h;
    if (!data || dataLen < 64) return -1;
    if (at_off < 0 || at_off + 64 > 64 * 1024 * 1024) return -2;
    unsigned long long off = (((unsigned long long)(unsigned)win_hi) << 32) | (unsigned)win_lo;
    off += (unsigned)at_off;
    int rc = rseek_to(off);
    if (rc) return 100 - rc;  // 110/111/112 = open/fwd-seek/back-seek
    rc = rread_full(data, 64);
    if (rc) return 120 - rc;  // 133 = read
    return 0;
}

static const unsigned char* g_cdsp_lm_bits = NULL;
static const short* g_cdsp_lm_scales = NULL;
static const unsigned char* g_cdsp_call_bits[256] = {0};
static const short* g_cdsp_call_scales[256] = {0};

int bonsai_gemv_q1r(remote_handle64 h, int out_dim, int in_dim, int prow,
                    int x_hi, int x_lo, int bits_hi, int bits_lo,
                    int scales_hi, int scales_lo, int y_hi, int y_lo) {
    (void)h;
    if (out_dim <= 0 || in_dim <= 0 || (in_dim & 127) != 0 || prow <= 0) return -2;
    const float* x = (const float*)(uintptr_t)join_ptr(x_hi, x_lo);
    const unsigned char* bits = (const unsigned char*)(uintptr_t)join_ptr(bits_hi, bits_lo);
    const short* scales = (const short*)(uintptr_t)join_ptr(scales_hi, scales_lo);
    float* y = (float*)(uintptr_t)join_ptr(y_hi, y_lo);
    return hvx_gemv_q1(out_dim, in_dim, prow, x, bits, scales, y);
}

static float g_dsp_signs_5120[5120] __attribute__((aligned(128)));
static int g_dsp_has_signs_5120 = 0;

int hvx_lmhead_static_q1(int out_dim, int in_dim, int prow,
                         const float* x, const unsigned char* bits,
                         const short* scales, const float* signs_5120, float* y);

int bonsai_gemv_q1(remote_handle64 h, int out_dim, int in_dim, int prow,
                   const float* x, int xLen,
                   const unsigned char* bits, int bitsLen,
                   const short* scales, int scalesLen, float* y, int yLen) {
    (void)h;
    if (!x || !bits || !scales || !y) return -1;
    // One-time static pointer registration for layer call c = 0..255
    if (out_dim <= -1000 && out_dim > -1256) {
        int c = -1000 - out_dim;
        g_cdsp_call_bits[c] = bits;
        g_cdsp_call_scales[c] = (in_dim > 0) ? (const short*)(bits + (size_t)(unsigned)in_dim) : scales;
        return 0;
    }
    // One-time static pointer registration for 303 MB LM head
    if (out_dim == 248320 && in_dim == 0) {
        g_cdsp_lm_bits = bits;
        g_cdsp_lm_scales = (prow > 0) ? (const short*)(bits + (size_t)(unsigned)prow) : scales;
        return 0;
    }
    int call_tag = (prow >> 16) & 0xFFFF;
    prow &= 0xFFFF;
    if (out_dim <= 0 || in_dim <= 0 || (in_dim & 127) != 0 || prow <= 0) return -2;
    if (xLen < in_dim) return -3;
    int batch = xLen / in_dim;
    int ng = in_dim >> 7;
    const unsigned char* use_bits = bits;
    const short* use_scales = scales;
    if (call_tag >= 1 && call_tag <= 256) {
        int c = call_tag - 1;
        if (!g_cdsp_call_bits[c] || !g_cdsp_call_scales[c]) return -4;
        use_bits = g_cdsp_call_bits[c];
        use_scales = g_cdsp_call_scales[c];
    } else if (out_dim == 248320 && bitsLen < out_dim * prow) {
        if (!g_cdsp_lm_bits || !g_cdsp_lm_scales) return -4;
        use_bits = g_cdsp_lm_bits;
        use_scales = g_cdsp_lm_scales;
    } else {
        if (bitsLen < out_dim * prow) return -4;
        if (scalesLen < out_dim * ng) return -5;
    }
    if (yLen < out_dim * batch) return -6;
    if (batch == 1) {
        if (call_tag == 999) {
            return hvx_lmhead_static_q1(out_dim, in_dim, prow, x, use_bits, use_scales,
                                        g_dsp_has_signs_5120 ? g_dsp_signs_5120 : NULL, y);
        }
        return hvx_gemv_q1(out_dim, in_dim, prow, x, use_bits, use_scales, y);
    } else {
        return hvx_gemm_q1(batch, out_dim, in_dim, prow, x, use_bits, use_scales, y);
    }
}

// ============================================================================
// Fused MLP Implementation: gate_up -> SwiGLU -> FWHT-1024 -> down_proj
// ============================================================================
static float g_dsp_signs_17408[17408] __attribute__((aligned(128)));
static int g_dsp_has_signs = 0;

static float g_dsp_mlp_gate_up[8 * 34816] __attribute__((aligned(128)));
static float g_dsp_mlp_fwht[8 * 17408] __attribute__((aligned(128)));

int bonsai_set_signs(remote_handle64 _h, const float* signs, int signsLen) {
    (void)_h;
    if (!signs || signsLen < 17408) return -1;
    memcpy(g_dsp_signs_17408, signs, 17408 * sizeof(float));
    g_dsp_has_signs = 1;
    return 0;
}

int bonsai_mlp_fused(remote_handle64 _h, const float* x, int xLen,
                     const unsigned char* gate_bits, int gate_bitsLen,
                     const short* gate_scales, int gate_scalesLen,
                     const unsigned char* down_bits, int down_bitsLen,
                     const short* down_scales, int down_scalesLen,
                     float* y, int yLen) {
    (void)_h;
    (void)gate_bitsLen; (void)gate_scalesLen;
    (void)down_bitsLen; (void)down_scalesLen;
    if (!x || !gate_bits || !gate_scales || !down_bits || !down_scales || !y) return -1;
    int batch = xLen / 5120;
    if (batch < 1 || batch > 8) return -2;
    if (yLen < batch * 5120) return -3;

    // Step 1: gate_up GEMV (out_dim=34816, in_dim=5120, prow=1280)
    int rc1 = (batch == 1) ?
        hvx_gemv_q1(34816, 5120, 1280, x, gate_bits, gate_scales, g_dsp_mlp_gate_up) :
        hvx_gemm_q1(batch, 34816, 5120, 1280, x, gate_bits, gate_scales, g_dsp_mlp_gate_up);
    if (rc1 != 0) return rc1;

    // Step 2 & 3: Parallel SwiGLU + FWHT-1024 across 6 QuRT threads
    int rc_mid = hvx_swiglu_fwht1024(batch, g_dsp_mlp_gate_up, g_dsp_has_signs ? g_dsp_signs_17408 : NULL, g_dsp_mlp_fwht);
    if (rc_mid != 0) return rc_mid;

    // Step 4: down_proj GEMV (out_dim=5120, in_dim=17408, prow=4352)
    int rc2 = (batch == 1) ?
        hvx_gemv_q1(5120, 17408, 4352, g_dsp_mlp_fwht, down_bits, down_scales, y) :
        hvx_gemm_q1(batch, 5120, 17408, 4352, g_dsp_mlp_fwht, down_bits, down_scales, y);
    if (rc2 != 0) return rc2;

    return 0;
}

// ============================================================================
// Fused Linear Attention Block Implementation
// ============================================================================
static float g_dsp_signs_6144[6144] __attribute__((aligned(128)));
static int g_dsp_has_signs_6144 = 0;

int g_dsp_kv_ctx_max = 256;

static float* g_dsp_ssm_conv = NULL;
static float* g_dsp_ssm_rec = NULL;
static float* g_dsp_ssm_conv_bak = NULL;
static float* g_dsp_ssm_rec_bak = NULL;
static unsigned char* g_dsp_lin_aux = NULL;
static void* g_dsp_kvk_raw = NULL;
static void* g_dsp_kvv_raw = NULL;
static uint32_t* g_dsp_kvk = NULL;
static uint32_t* g_dsp_kvv = NULL;
static float* g_dsp_kvks = NULL;
static float* g_dsp_kvvs = NULL;

int hvx_lin_attn_fused(int layer_idx,
                       const float* x,
                       const unsigned char* in_bits, const short* in_scales,
                       const unsigned char* out_bits, const short* out_scales,
                       const float* ssm_conv, const float* ssm_rec, const uint8_t* lin_aux,
                       const float* signs_5120, const float* signs_6144,
                       float* y);

int hvx_lin_layer_fused(int layer_idx,
                        const float* x,
                        const unsigned char* in_bits, const short* in_scales,
                        const unsigned char* out_bits, const short* out_scales,
                        const unsigned char* gate_bits, const short* gate_scales,
                        const unsigned char* down_bits, const short* down_scales,
                        const float* ssm_conv, const float* ssm_rec, const uint8_t* lin_aux,
                        uint32_t* kvk, uint32_t* kvv, float* kvks, float* kvvs,
                        const float* signs_5120, const float* signs_6144, const float* signs_17408,
                        float* mlp_gate_up_buf, float* mlp_fwht_buf,
                        float* y);

int bonsai_set_signs_dim(remote_handle64 _h, int dim, const float* signs, int signsLen) {
    (void)_h;
    if (dim == 5120 && signs && signsLen >= 5120) {
        memcpy(g_dsp_signs_5120, signs, 5120 * sizeof(float));
        g_dsp_has_signs_5120 = 1;
        return 0;
    }
    if (dim == 6144 && signs && signsLen >= 6144) {
        memcpy(g_dsp_signs_6144, signs, 6144 * sizeof(float));
        g_dsp_has_signs_6144 = 1;
        return 0;
    }
    if (dim == 17408 && signs && signsLen >= 17408) {
        memcpy(g_dsp_signs_17408, signs, 17408 * sizeof(float));
        g_dsp_has_signs = 1;
        return 0;
    }
    return -1;
}

int bonsai_register_lin_states(remote_handle64 _h,
                               const float* ssm_conv, int ssm_convLen,
                               const float* ssm_rec, int ssm_recLen,
                               const unsigned char* lin_aux, int lin_auxLen) {
    (void)_h; (void)ssm_conv; (void)ssm_convLen; (void)ssm_rec; (void)ssm_recLen;
    size_t conv_bytes = (size_t)48 * 10240 * 3 * sizeof(float);
    size_t rec_bytes  = (size_t)48 * 48 * 128 * 128 * sizeof(float);

    // Fast in-place reset for multi-turn chat / server requests
    if (!lin_aux || lin_auxLen == 0) {
        if (g_dsp_ssm_conv) memset(g_dsp_ssm_conv, 0, conv_bytes);
        if (g_dsp_ssm_rec)  memset(g_dsp_ssm_rec, 0, rec_bytes);
        if (g_dsp_kvk && g_dsp_kv_ctx_max > 0) {
            size_t kv_bytes  = (size_t)16 * 4 * g_dsp_kv_ctx_max * 128 * sizeof(uint32_t);
            size_t kvs_bytes = (size_t)16 * 4 * g_dsp_kv_ctx_max * sizeof(float);
            memset(g_dsp_kvk, 0, kv_bytes);
            memset(g_dsp_kvv, 0, kv_bytes);
            memset(g_dsp_kvks, 0, kvs_bytes);
            memset(g_dsp_kvvs, 0, kvs_bytes);
        }
        return 0;
    }

    size_t base_aux = (size_t)48 * 1188736 + (size_t)16 * 43008;
    int req_ctx = 256;
    if (lin_aux && (size_t)lin_auxLen > base_aux) {
        req_ctx = (int)(((size_t)lin_auxLen - base_aux) / 256);
    }
    if (req_ctx < 256) req_ctx = 256;
    if (!g_dsp_ssm_conv) {
        g_dsp_ssm_conv = (float*)malloc(conv_bytes);
        if (!g_dsp_ssm_conv) return -11;
    }
    memset(g_dsp_ssm_conv, 0, conv_bytes);

    if (!g_dsp_ssm_rec) {
        g_dsp_ssm_rec = (float*)malloc(rec_bytes);
        if (!g_dsp_ssm_rec) return -12;
    }
    memset(g_dsp_ssm_rec, 0, rec_bytes);

    if (!g_dsp_ssm_conv_bak) {
        g_dsp_ssm_conv_bak = (float*)malloc(conv_bytes);
    }
    if (!g_dsp_ssm_rec_bak) {
        g_dsp_ssm_rec_bak = (float*)malloc(rec_bytes);
    }

    // Free previously allocated KV buffers if re-registering
    if (g_dsp_kvk_raw) { free(g_dsp_kvk_raw); g_dsp_kvk_raw = NULL; g_dsp_kvk = NULL; }
    if (g_dsp_kvv_raw) { free(g_dsp_kvv_raw); g_dsp_kvv_raw = NULL; g_dsp_kvv = NULL; }
    if (g_dsp_kvks)    { free(g_dsp_kvks);    g_dsp_kvks = NULL; }
    if (g_dsp_kvvs)    { free(g_dsp_kvvs);    g_dsp_kvvs = NULL; }

    int target_ctx = req_ctx;
    while (target_ctx >= 256) {
        size_t kv_bytes   = (size_t)16 * 4 * target_ctx * 128 * sizeof(uint32_t);
        size_t kvs_bytes  = (size_t)16 * 4 * target_ctx * sizeof(float);

        g_dsp_kvk_raw = malloc(kv_bytes + 128);
        g_dsp_kvv_raw = malloc(kv_bytes + 128);
        g_dsp_kvks    = (float*)malloc(kvs_bytes);
        g_dsp_kvvs    = (float*)malloc(kvs_bytes);

        if (g_dsp_kvk_raw && g_dsp_kvv_raw && g_dsp_kvks && g_dsp_kvvs) {
            g_dsp_kvk = (uint32_t*)(((uintptr_t)g_dsp_kvk_raw + 127u) & ~(uintptr_t)127u);
            g_dsp_kvv = (uint32_t*)(((uintptr_t)g_dsp_kvv_raw + 127u) & ~(uintptr_t)127u);
            g_dsp_kv_ctx_max = target_ctx;
            memset(g_dsp_kvk, 0, kv_bytes);
            memset(g_dsp_kvv, 0, kv_bytes);
            memset(g_dsp_kvks, 0, kvs_bytes);
            memset(g_dsp_kvvs, 0, kvs_bytes);
            break;
        }

        if (g_dsp_kvk_raw) { free(g_dsp_kvk_raw); g_dsp_kvk_raw = NULL; }
        if (g_dsp_kvv_raw) { free(g_dsp_kvv_raw); g_dsp_kvv_raw = NULL; }
        if (g_dsp_kvks)    { free(g_dsp_kvks);    g_dsp_kvks = NULL; }
        if (g_dsp_kvvs)    { free(g_dsp_kvvs);    g_dsp_kvvs = NULL; }

        if (target_ctx == 256) return -14;
        target_ctx /= 2;
        if (target_ctx < 256) target_ctx = 256;
    }

    size_t aux_bytes = (size_t)48 * 1188736 + (size_t)16 * 43008 + (size_t)g_dsp_kv_ctx_max * 256;
    if (lin_aux && lin_auxLen >= (int)((size_t)48 * 1147776)) {
        if (g_dsp_lin_aux) {
            free(g_dsp_lin_aux);
            g_dsp_lin_aux = NULL;
        }
        g_dsp_lin_aux = (unsigned char*)malloc(aux_bytes);
        if (!g_dsp_lin_aux) return -13;
        size_t copy_bytes = (size_t)lin_auxLen < aux_bytes ? (size_t)lin_auxLen : aux_bytes;
        memcpy(g_dsp_lin_aux, lin_aux, copy_bytes);
    }

    return 0;
}

int bonsai_lin_attn_fused(remote_handle64 _h, int layer_idx,
                          const float* x, int xLen,
                          const unsigned char* in_bits, int in_bitsLen,
                          const short* in_scales, int in_scalesLen,
                          const unsigned char* out_bits, int out_bitsLen,
                          const short* out_scales, int out_scalesLen,
                          float* y, int yLen) {
    (void)_h; (void)xLen;
    (void)in_bitsLen; (void)in_scalesLen;
    (void)out_bitsLen; (void)out_scalesLen;
    (void)yLen;
    if (!x || !in_bits || !in_scales || !y) return -1;
    if (!g_dsp_ssm_conv || !g_dsp_ssm_rec || !g_dsp_lin_aux) return -2;
    if (layer_idx < -16 || layer_idx >= 48) return -3;

    return hvx_lin_attn_fused(layer_idx, x,
                              in_bits, in_scales,
                              out_bits, out_scales,
                              g_dsp_ssm_conv, g_dsp_ssm_rec, g_dsp_lin_aux,
                              g_dsp_has_signs_5120 ? g_dsp_signs_5120 : NULL,
                              g_dsp_has_signs_6144 ? g_dsp_signs_6144 : NULL,
                              y);
}

int bonsai_lin_layer_fused(remote_handle64 _h, int layer_idx,
                           const float* x, int xLen,
                           const unsigned char* in_bits, int in_bitsLen,
                           const short* in_scales, int in_scalesLen,
                           const unsigned char* out_bits, int out_bitsLen,
                           const short* out_scales, int out_scalesLen,
                           const unsigned char* gate_bits, int gate_bitsLen,
                           const short* gate_scales, int gate_scalesLen,
                           const unsigned char* down_bits, int down_bitsLen,
                           const short* down_scales, int down_scalesLen,
                           float* y, int yLen) {
    (void)_h; (void)xLen;
    (void)in_bitsLen; (void)in_scalesLen;
    (void)out_bitsLen; (void)out_scalesLen;
    (void)gate_bitsLen; (void)gate_scalesLen;
    (void)down_bitsLen; (void)down_scalesLen;
    (void)yLen;
    if (layer_idx == -500) {
        if (!g_dsp_ssm_conv || !g_dsp_ssm_rec || !g_dsp_ssm_conv_bak || !g_dsp_ssm_rec_bak) return -500;
        memcpy(g_dsp_ssm_conv_bak, g_dsp_ssm_conv, (size_t)48 * 10240 * 3 * sizeof(float));
        memcpy(g_dsp_ssm_rec_bak,  g_dsp_ssm_rec,  (size_t)48 * 48 * 128 * 128 * sizeof(float));
        return 0;
    }
    if (layer_idx == -501) {
        if (!g_dsp_ssm_conv || !g_dsp_ssm_rec || !g_dsp_ssm_conv_bak || !g_dsp_ssm_rec_bak) return -501;
        memcpy(g_dsp_ssm_conv, g_dsp_ssm_conv_bak, (size_t)48 * 10240 * 3 * sizeof(float));
        memcpy(g_dsp_ssm_rec,  g_dsp_ssm_rec_bak,  (size_t)48 * 48 * 128 * 128 * sizeof(float));
        return 0;
    }
    if (!x || !out_bits || !out_scales ||
        !gate_bits || !gate_scales || !down_bits || !down_scales || !y) return -1;
    if (!g_dsp_ssm_conv || !g_dsp_ssm_rec || !g_dsp_lin_aux) return -2;
    if ((layer_idx < -16 && layer_idx != -100 && layer_idx != -101 &&
         !(layer_idx <= -200 && layer_idx >= -203) &&
         !(layer_idx <= -10000 && layer_idx > -10000 - 16 * g_dsp_kv_ctx_max)) || layer_idx >= 48) return -3;

    return hvx_lin_layer_fused(layer_idx, x,
                               in_bits, in_scales,
                               out_bits, out_scales,
                               gate_bits, gate_scales,
                               down_bits, down_scales,
                               g_dsp_ssm_conv, g_dsp_ssm_rec, g_dsp_lin_aux,
                               g_dsp_kvk, g_dsp_kvv, g_dsp_kvks, g_dsp_kvvs,
                               g_dsp_has_signs_5120 ? g_dsp_signs_5120 : NULL,
                               g_dsp_has_signs_6144 ? g_dsp_signs_6144 : NULL,
                               g_dsp_has_signs ? g_dsp_signs_17408 : NULL,
                               g_dsp_mlp_gate_up, g_dsp_mlp_fwht,
                               y);
}


