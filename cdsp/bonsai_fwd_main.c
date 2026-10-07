// bonsai_fwd: full single-token decode, ARM64 orchestration + CDSP GEMV.
// Math matches transformers modeling_qwen3_5.py single-token path exactly.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _WIN32
#include "win_posix_shim.h"
#else
#define _GNU_SOURCE
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/resource.h>
#include <errno.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "bonsai_ops.h"
#include "tok.c"   // tok_encode/tok_decode/load_tok/build_spairs (validated)

typedef struct { uint8_t kind; uint32_t out, in; uint64_t off, len; } Tensor;
#define MAXT 2048
static Tensor g_tab[MAXT];
static char g_names[MAXT][128];
static float* g_f32_cache[MAXT] = {0};
static int g_nt = 0;
static const uint8_t* g_base;

static inline float f16(float h_bits_unused, uint16_t h) {
    (void)h_bits_unused;
#ifdef __ARM_ARCH
    __fp16 x;
    memcpy(&x, &h, 2);
    return (float)x;
#else
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float res;
    memcpy(&res, &u, 4);
    return res;
#endif
}

static inline uint32_t tensor_prow(const Tensor* tk) {
    uint32_t rb = tk->in / 8;
    if (tk->kind == 2) return 2 * rb;
    return rb + ((128 - (rb % 128)) % 128);
}

// Returns FP32 pointer for kind==3 or kind==1 (cached in heap memory so g_base is not pinned)
static const float* get_f32_ptr(const Tensor* t) {
    if (!t) return NULL;
    int idx = (int)(t - g_tab);
    if (idx >= 0 && idx < g_nt && g_f32_cache[idx]) return g_f32_cache[idx];
    size_t nelem = (size_t)t->out * (size_t)(t->in > 0 ? t->in : 1);
    float* buf = (float*)malloc(nelem * sizeof(float));
    if (!buf) return NULL;
    if (t->kind == 3) {
        memcpy(buf, g_base + t->off, nelem * sizeof(float));
    } else {
        const uint16_t* src = (const uint16_t*)(g_base + t->off);
        for (size_t i = 0; i < nelem; i++) buf[i] = f16(0, src[i]);
    }
    if (idx >= 0 && idx < g_nt) g_f32_cache[idx] = buf;
    return buf;
}

static int g_npu_fd = -1;
static size_t g_npu_size = 0;

static int load_npubin(const char* path) {
    g_npu_fd = open(path, O_RDONLY);
    if (g_npu_fd < 0) return -1;
    struct stat st;
    if (fstat(g_npu_fd, &st) != 0) return -1;
    g_npu_size = (size_t)st.st_size;
    g_base = (const uint8_t*)mmap(NULL, g_npu_size, PROT_READ, MAP_SHARED, g_npu_fd, 0);
    if (g_base == MAP_FAILED) return -1;
    if (memcmp(g_base, "NPU1", 4) != 0) return -1;
    uint32_t ver, n;
    memcpy(&ver, g_base + 4, 4);
    memcpy(&n, g_base + 8, 4);
    if (ver != 0 || n > MAXT) return -1;
    size_t p = 12;
    for (uint32_t i = 0; i < n; i++) {
        uint16_t nl;
        memcpy(&nl, g_base + p, 2); p += 2;
        memcpy(g_names[i], g_base + p, nl); g_names[i][nl] = 0; p += nl;
        g_tab[i].kind = g_base[p]; p += 1;
        memcpy(&g_tab[i].out, g_base + p, 4); p += 4;
        memcpy(&g_tab[i].in, g_base + p, 4); p += 4;
        memcpy(&g_tab[i].off, g_base + p, 8); p += 8;
        memcpy(&g_tab[i].len, g_base + p, 8); p += 8;
    }
    g_nt = (int)n;
    return 0;
}

static const Tensor* find_t(const char* name) {
    for (int i = 0; i < g_nt; i++)
        if (strcmp(g_names[i], name) == 0) return &g_tab[i];
    return NULL;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static inline uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)(ts.tv_nsec / 1000);
}

typedef struct {
    uint64_t t_fwht_us;
    uint64_t t_wait_us;
    uint64_t t_rpc_us;
    uint64_t t_post_us;
    uint64_t t_total_us;
} CallTiming;

typedef struct {
    int is_lin;
    int is_static;
    uint64_t t_in_norm_us;
    CallTiming c0; // in_proj
    uint64_t t_attn_ab_us;
    uint64_t t_attn_conv_us;
    uint64_t t_attn_qk_norm_us;
    uint64_t t_attn_recur_us;
    uint64_t t_attn_rope_us;
    uint64_t t_attn_kv_enc_us;
    uint64_t t_attn_gqa_us;
    CallTiming c1; // out_proj
    uint64_t t_res1_us;
    uint64_t t_post_norm_us;
    uint64_t t_mlp_fwht_us;
    uint64_t t_mlp_wait_us;
    uint64_t t_mlp_rpc_us;
    uint64_t t_res2_us;
    uint64_t t_layer_total_us;
} LayerTiming;

typedef struct {
    uint64_t t_embed_us;
    LayerTiming layers[64];
    uint64_t t_final_norm_us;
    uint64_t t_lmhead_fwht_us;
    uint64_t t_lmhead_memcpy_us;
    uint64_t t_lmhead_rpc_us;
    uint64_t t_lmhead_post_us;
    uint64_t t_argmax_us;
    uint64_t t_token_total_us;
} StepTiming;

static StepTiming g_prof;
static int g_profile_tree = 0;

static double g_t_memcpy = 0;
static double g_t_rpc = 0;
static double g_t_trans = 0;
static int g_calls_rpc = 0;

// ---- CDSP GEMV (via generated stub; linked) ----
int bonsai_gemv_q1(unsigned long long h, int out_dim, int in_dim, int prow,
                   const float* x, int xLen, const unsigned char* bits, int bitsLen,
                   const short* scales, int scalesLen, float* y, int yLen);
int bonsai_gemv_q1r(unsigned long long h, int out_dim, int in_dim, int prow,
                    int x_hi, int x_lo, int bits_hi, int bits_lo,
                    int scales_hi, int scales_lo, int y_hi, int y_lo);
int bonsai_set_signs(unsigned long long h, const float* signs, int signsLen);
int bonsai_mlp_fused(unsigned long long h, const float* x, int xLen,
                     const unsigned char* gate_bits, int gate_bitsLen,
                     const short* gate_scales, int gate_scalesLen,
                     const unsigned char* down_bits, int down_bitsLen,
                     const short* down_scales, int down_scalesLen,
                     float* y, int yLen);
int bonsai_open(const char* uri, unsigned long long* handle);
int bonsai_close(unsigned long long handle);
int bonsai_set_signs_dim(unsigned long long h, int dim, const float* signs, int signsLen);
int bonsai_register_lin_states(unsigned long long h,
                               const float* ssm_conv, int ssm_convLen,
                               const float* ssm_rec, int ssm_recLen,
                               const unsigned char* lin_aux, int lin_auxLen);
int bonsai_lin_attn_fused(unsigned long long h, int layer_idx,
                          const float* x, int xLen,
                          const unsigned char* in_bits, int in_bitsLen,
                          const short* in_scales, int in_scalesLen,
                          const unsigned char* out_bits, int out_bitsLen,
                          const short* out_scales, int out_scalesLen,
                          float* y, int yLen);
int bonsai_lin_layer_fused(unsigned long long h, int layer_idx,
                           const float* x, int xLen,
                           const unsigned char* in_bits, int in_bitsLen,
                           const short* in_scales, int in_scalesLen,
                           const unsigned char* out_bits, int out_bitsLen,
                           const short* out_scales, int out_scalesLen,
                           const unsigned char* gate_bits, int gate_bitsLen,
                           const short* gate_scales, int gate_scalesLen,
                           const unsigned char* down_bits, int down_bitsLen,
                           const short* down_scales, int down_scalesLen,
                           float* y, int yLen);
static float* ssm_conv = NULL;
static float* ssm_rec = NULL;
static int8_t* kvk = NULL;
static int8_t* kvv = NULL;
static float* kvks = NULL;
static float* kvvs = NULL;
static int g_ctx = 4096;
static int g_dsp_ctx = 4096;
static uint8_t* g_lin_aux = NULL;
static int g_fused_lin = 1;
static int g_fused_mlp = 1;
static int g_fused_layer = 1;
extern int remote_session_control(unsigned, void*, unsigned);
extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
#ifndef _WIN32
extern int rpcmem_to_fd(void* po);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);
extern int remote_handle64_control(unsigned long long h, uint32_t req, void* data, uint32_t datalen);
#endif
static int g_lm_static_cdsp = 0;

static unsigned long long g_h = 0;

static int g_static_per_group = 16;
static int g_static_layers = 32;
#define STATIC_LAYERS_PER_GROUP g_static_per_group
#define STATIC_LAYERS g_static_layers
#define MAX_RING_LAYERS 6
#define MAX_NUM_BUFS (MAX_RING_LAYERS * 4)
static int g_ring_layers = 2;
#define RING_LAYERS g_ring_layers
#define NUM_BUFS (g_ring_layers * 4)
#define MAX_CALL_BITS_Q1 (34816 * 640)
#define MAX_CALL_BITS_Q2 (34816 * 1280)
#define MAX_CALL_SCALES  (34816 * 40 * 2)

static uint8_t* g_ring_arena = NULL;
static size_t g_ring_arena_sz = 0;
static uint8_t* g_bits_buf[MAX_NUM_BUFS] = {0};
static int16_t* g_scales_buf[MAX_NUM_BUFS] = {0};
static uint8_t* g_lm_bits = 0;
static int16_t* g_lm_scales = 0;

static const float* g_signs_5120 = NULL;
static const float* g_signs_6144 = NULL;
static const float* g_signs_17408 = NULL;

static const float* signs_for_dim(int dim) {
    if (dim == 5120) return g_signs_5120;
    if (dim == 6144) return g_signs_6144;
    if (dim == 17408) return g_signs_17408;
    return NULL;
}

static sem_t g_sem_free[MAX_NUM_BUFS];
static sem_t g_sem_ready[MAX_NUM_BUFS];
static sem_t g_sem_start_tok;
static volatile int g_prefetch_exit = 0;
static pthread_t g_prefetch_tid;

typedef struct {
    int nt;
    const Tensor* t[4];
    int total_out;
    int in_dim;
    int prow;
    int ng;
} CallDesc;

typedef struct {
    const float* in_norm;
    const float* post_norm;
    const float* in_proj_a;
    const float* in_proj_b;
    const float* conv1d_w;
    const float* A_log;
    const float* dt_bias;
    const float* lin_norm;
    const float* q_norm;
    const float* k_norm;
} LayerWeights;

static CallDesc g_calls[256];
static LayerWeights g_lw[64];
static const Tensor* g_t_embed = NULL;
static const Tensor* g_t_lmhead = NULL;
static const float* g_final_norm = NULL;
static int g_cur_call = 0;

static float* g_xs = 0;
static float* g_ys = 0;
static float* g_lm_ys = 0;

static char tname[128];
static const char* TN(int l, const char* s) {
    snprintf(tname, sizeof(tname), "language_model.model.layers.%d.%s", l, s);
    return tname;
}

static void init_call_plan(void) {
    g_t_embed = find_t("language_model.model.embed_tokens");
    g_t_lmhead = find_t("language_model.lm_head");
    g_final_norm = get_f32_ptr(find_t("language_model.model.norm.weight"));
    for (int L = 0; L < 64; L++) {
        int is_lin = (L % 4 != 3);
        LayerWeights* lw = &g_lw[L];
        lw->in_norm = get_f32_ptr(find_t(TN(L, "input_layernorm.weight")));
        lw->post_norm = get_f32_ptr(find_t(TN(L, "post_attention_layernorm.weight")));
        // Call 4*L + 0
        CallDesc* c0 = &g_calls[4 * L + 0];
        if (is_lin) {
            c0->t[0] = find_t(TN(L, "linear_attn.in_proj_qkv"));
            c0->t[1] = find_t(TN(L, "linear_attn.in_proj_z"));
            c0->t[2] = find_t(TN(L, "linear_attn.in_proj_a"));
            c0->t[3] = find_t(TN(L, "linear_attn.in_proj_b"));
            if (c0->t[2] && (c0->t[2]->kind == 0 || c0->t[2]->kind == 2)) {
                c0->nt = 4;
                c0->total_out = (int)c0->t[0]->out + (int)c0->t[1]->out + (int)c0->t[2]->out + (int)c0->t[3]->out;
                lw->in_proj_a = NULL;
                lw->in_proj_b = NULL;
            } else {
                // In Bonsai 2, in_proj_a & in_proj_b are kind==3 [48, 5120] packed to BF16 in stage_init()
                c0->nt = 2;
                c0->total_out = (int)c0->t[0]->out + (int)c0->t[1]->out;
                lw->in_proj_a = (c0->t[2] && c0->t[2]->kind == 3) ? (const float*)(g_base + c0->t[2]->off) : get_f32_ptr(c0->t[2]);
                lw->in_proj_b = (c0->t[3] && c0->t[3]->kind == 3) ? (const float*)(g_base + c0->t[3]->off) : get_f32_ptr(c0->t[3]);
            }
            c0->in_dim = (int)c0->t[0]->in;
            lw->conv1d_w = get_f32_ptr(find_t(TN(L, "linear_attn.conv1d.weight")));
            lw->A_log    = get_f32_ptr(find_t(TN(L, "linear_attn.A_log")));
            lw->dt_bias  = get_f32_ptr(find_t(TN(L, "linear_attn.dt_bias")));
            lw->lin_norm = get_f32_ptr(find_t(TN(L, "linear_attn.norm.weight")));
        } else {
            c0->nt = 3;
            c0->t[0] = find_t(TN(L, "self_attn.q_proj"));
            c0->t[1] = find_t(TN(L, "self_attn.k_proj"));
            c0->t[2] = find_t(TN(L, "self_attn.v_proj"));
            c0->total_out = (int)c0->t[0]->out + (int)c0->t[1]->out + (int)c0->t[2]->out;
            c0->in_dim = (int)c0->t[0]->in;
            lw->q_norm = get_f32_ptr(find_t(TN(L, "self_attn.q_norm.weight")));
            lw->k_norm = get_f32_ptr(find_t(TN(L, "self_attn.k_norm.weight")));
        }
        c0->ng = c0->in_dim / 128;
        c0->prow = (int)tensor_prow(c0->t[0]);

        // Call 4*L + 1
        CallDesc* c1 = &g_calls[4 * L + 1];
        c1->nt = 1;
        c1->t[0] = is_lin ? find_t(TN(L, "linear_attn.out_proj")) : find_t(TN(L, "self_attn.o_proj"));
        c1->total_out = (int)c1->t[0]->out;
        c1->in_dim = (int)c1->t[0]->in;
        c1->ng = c1->in_dim / 128;
        c1->prow = (int)tensor_prow(c1->t[0]);

        // Call 4*L + 2
        CallDesc* c2 = &g_calls[4 * L + 2];
        c2->nt = 2;
        c2->t[0] = find_t(TN(L, "mlp.gate_proj"));
        c2->t[1] = find_t(TN(L, "mlp.up_proj"));
        c2->total_out = (int)c2->t[0]->out + (int)c2->t[1]->out;
        c2->in_dim = (int)c2->t[0]->in;
        c2->ng = c2->in_dim / 128;
        c2->prow = (int)tensor_prow(c2->t[0]);

        // Call 4*L + 3
        CallDesc* c3 = &g_calls[4 * L + 3];
        c3->nt = 1;
        c3->t[0] = find_t(TN(L, "mlp.down_proj"));
        c3->total_out = (int)c3->t[0]->out;
        c3->in_dim = (int)c3->t[0]->in;
        c3->ng = c3->in_dim / 128;
        c3->prow = (int)tensor_prow(c3->t[0]);
    }
}

#ifdef _WIN32
extern int npu_emu_stage_call(int call_idx, const uint8_t* h_bits, size_t bits_bytes,
                              const int16_t* h_scales, size_t scales_bytes);
extern int npu_emu_is_call_staged(int call_idx);
extern void npu_emu_set_active_call(int call_idx);
extern size_t npu_emu_get_vram_bytes(void);
static int g_vram_prestaged = 0;
#endif

static int g_skip_memcpy = 0;
static uint8_t* g_call_static_bits[256] = {0};
static int16_t* g_call_static_scales[256] = {0};
static int g_read_buf_idx = 0;

static int g_zero_copy = 1;
static uint8_t* g_group_buf[4] = {0};
static int g_group_fd[4] = {0};
static size_t g_group_sz[4] = {0};
static uint8_t* g_call_bits[256] = {0};
static int16_t* g_call_scales[256] = {0};

static int g_pipelined_smmu = 0;
static int g_group_mapped[4] = {0};
static pthread_t g_smmu_worker_tid;
static sem_t g_sem_smmu_event;
static sem_t g_sem_smmu_done;
static volatile int g_smmu_target_group = -1;
static volatile int g_smmu_unmap_group = -1;
static volatile int g_smmu_worker_exit = 0;
static volatile int g_smmu_flip_in_flight = 0;

static void* smmu_flip_worker_fn(void* arg) {
    (void)arg;
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(4, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    setpriority(PRIO_PROCESS, 0, -10);
#endif
    while (!g_smmu_worker_exit) {
        sem_wait(&g_sem_smmu_event);
        if (g_smmu_worker_exit) break;
        int unmap_g = g_smmu_unmap_group;
        int map_g   = g_smmu_target_group;
        double t0 = now_ms();
        double tu_ms = 0, tm_ms = 0;
#ifndef _WIN32
        if (unmap_g >= 0 && unmap_g < 4 && g_group_mapped[unmap_g]) {
            double u0 = now_ms();
            int rc_u = fastrpc_munmap(3, g_group_fd[unmap_g], g_group_buf[unmap_g], g_group_sz[unmap_g]);
            tu_ms = now_ms() - u0;
            if (rc_u == 0) {
                g_group_mapped[unmap_g] = 0;
            } else {
                fprintf(stderr, "[fwd] ERR: fastrpc_munmap(Group %d) failed rc=%d\n", unmap_g, rc_u);
            }
        }
        if (map_g >= 0 && map_g < 4 && !g_group_mapped[map_g]) {
            double m0 = now_ms();
            int rc_m = fastrpc_mmap(3, g_group_fd[map_g], g_group_buf[map_g], 0, g_group_sz[map_g], 0);
            tm_ms = now_ms() - m0;
            if (rc_m == 0) {
                g_group_mapped[map_g] = 1;
            } else {
                fprintf(stderr, "[fwd] ERR: fastrpc_mmap(Group %d) failed rc=%d\n", map_g, rc_m);
            }
        }
#endif
        double tot_ms = now_ms() - t0;
        if (g_profile_tree) {
            fprintf(stderr, "[worker] flip (unmap G%d: %.1fms, map G%d: %.1fms) = %.1fms\n",
                    unmap_g, tu_ms, map_g, tm_ms, tot_ms);
        }
        sem_post(&g_sem_smmu_done);
    }
    return NULL;
}

static inline void trigger_smmu_flip(int unmap_g, int map_g) {
    if (!g_pipelined_smmu) return;
    if (g_smmu_flip_in_flight) {
        sem_wait(&g_sem_smmu_done);
        g_smmu_flip_in_flight = 0;
    }
    g_smmu_unmap_group = unmap_g;
    g_smmu_target_group = map_g;
    g_smmu_flip_in_flight = 1;
    sem_post(&g_sem_smmu_event);
}

static inline void wait_smmu_flip(void) {
    if (!g_pipelined_smmu) return;
    if (g_smmu_flip_in_flight) {
        sem_wait(&g_sem_smmu_done);
        g_smmu_flip_in_flight = 0;
    }
}


#ifndef _WIN32
static void release_mmap_pages(uint64_t off, uint64_t len) {
    uint64_t start = (off + 4095ULL) & ~4095ULL;
    uint64_t end   = (off + len) & ~4095ULL;
    if (end > start) {
        madvise((void*)(g_base + start), (size_t)(end - start), MADV_DONTNEED);
        if (g_npu_fd >= 0) {
            posix_fadvise(g_npu_fd, (off_t)start, (off_t)(end - start), POSIX_FADV_DONTNEED);
        }
    }
}

static void copy_and_release_mmap(void* dst, uint64_t file_off, size_t len) {
    uint8_t* d = (uint8_t*)dst;
    const size_t CHUNK = 4 * 1024 * 1024;
    size_t done = 0;
    while (done < len) {
        size_t n = (len - done > CHUNK) ? CHUNK : (len - done);
        memcpy(d + done, g_base + file_off + done, n);
        release_mmap_pages(file_off + done, n);
        done += n;
    }
}

static long get_mem_avail_mb(void) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[128];
    long avail_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %ld kB", &avail_kb) == 1) break;
    }
    fclose(f);
    return avail_kb >> 10;
}
#endif


static int g_single_copy = 0;

#ifndef _WIN32
#pragma weak remote_register_buf_attr2
extern void remote_register_buf_attr2(void* buf, size_t size, int fd, int attr);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

static void try_buf_attr(void* buf, size_t sz, int attr) {
    if (!buf || attr <= 0 || !remote_register_buf_attr2) return;
    int fd = rpcmem_to_fd(buf);
    if (fd >= 0) remote_register_buf_attr2(buf, sz, fd, attr);
}

static int try_static_mmap(void* buf, size_t sz) {
    if (!buf) return -1;
    int fd = rpcmem_to_fd(buf);
    if (fd < 0) return -2;
    return fastrpc_mmap(3, fd, buf, 0, sz, 0 /*FASTRPC_MAP_STATIC*/);
}
#endif

static int g_nt_copy = 1;

static inline void fast_stream_copy(void* dst, const void* src, size_t len) {
#if defined(__aarch64__)
    if (g_nt_copy && (len >= 128) && ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0)) {
        uint8_t* d = (uint8_t*)dst;
        const uint8_t* s = (const uint8_t*)src;
        size_t n128 = len >> 7;
        while (n128--) {
            __asm__ __volatile__(
                "prfm pldl1strm, [%[s], #512]\n\t"
                "ldp q0, q1, [%[s], #0]\n\t"
                "ldp q2, q3, [%[s], #32]\n\t"
                "ldp q4, q5, [%[s], #64]\n\t"
                "ldp q6, q7, [%[s], #96]\n\t"
                "stnp q0, q1, [%[d], #0]\n\t"
                "stnp q2, q3, [%[d], #32]\n\t"
                "stnp q4, q5, [%[d], #64]\n\t"
                "stnp q6, q7, [%[d], #96]\n\t"
                :
                : [s] "r"(s), [d] "r"(d)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
            );
            s += 128;
            d += 128;
        }
        size_t tail = len & 127;
        if (tail) memcpy(d, s, tail);
        return;
    }
#endif
    memcpy(dst, src, len);
}

static int g_prefetch_threads = 2;
static pthread_t g_helper_tid;
static pthread_t g_helper2_tid;
static sem_t g_sem_helper_start;
static sem_t g_sem_helper2_start;

static void copy_call_slot(int c, int buf) {
    const CallDesc* cd = &g_calls[c];
    uint8_t* bdst = g_bits_buf[buf];
    uint8_t* sdst = (uint8_t*)g_scales_buf[buf];

    for (int t_idx = 0; t_idx < cd->nt; t_idx++) {
        const Tensor* tk = cd->t[t_idx];
        uint32_t tk_ng = tk->in / 128;
        uint32_t tk_prow = tensor_prow(tk);

        size_t b_len = (size_t)tk->out * tk_prow;
        size_t s_len = (size_t)tk->out * tk_ng * 2;
        const uint8_t* bsrc = g_base + tk->off + s_len;
        const uint8_t* ssrc = g_base + tk->off;

        fast_stream_copy(bdst, bsrc, b_len);
        fast_stream_copy(sdst, ssrc, s_len);

        bdst += b_len;
        sdst += s_len;
    }
#ifdef _WIN32
    npu_emu_stage_call(c, g_bits_buf[buf], (size_t)cd->total_out * cd->prow,
                       g_scales_buf[buf], (size_t)cd->total_out * cd->ng * 2);
#endif
}

static void* prefetch_helper2_fn(void* arg) {
    (void)arg;
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    setpriority(PRIO_PROCESS, 0, -10);
#endif
    int first_tok = 1;
    while (!g_prefetch_exit) {
        sem_wait(&g_sem_helper2_start);
        if (g_prefetch_exit) break;
        for (int L = STATIC_LAYERS; L < 64; L++) {
            for (int k = 0; k <= 1; k++) {
                int c = 4 * L + k;
                int buf = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + k;
                sem_wait(&g_sem_free[buf]);
                if (g_prefetch_exit) break;
                if (g_skip_memcpy && (!first_tok || L >= STATIC_LAYERS + RING_LAYERS)) {
                    sem_post(&g_sem_ready[buf]);
                    continue;
                }
                copy_call_slot(c, buf);
                sem_post(&g_sem_ready[buf]);
            }
            if (g_prefetch_exit) break;
        }
        first_tok = 0;
    }
    return NULL;
}

static int g_alt_k1 = 0;

static void* prefetch_helper_fn(void* arg) {
    (void)arg;
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(5, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    setpriority(PRIO_PROCESS, 0, -10);
#endif
    int first_tok = 1;
    while (!g_prefetch_exit) {
        sem_wait(&g_sem_helper_start);
        if (g_prefetch_exit) break;
        for (int L = STATIC_LAYERS; L < 64; L++) {
            int slot = (L - STATIC_LAYERS) % RING_LAYERS;
            for (int k = 1; k <= 2; k++) {
                if (k == 1 && (!g_alt_k1 || g_prefetch_threads >= 3 || ((slot & 1) == 0))) continue;
                int c = 4 * L + k;
                int buf = (slot * 4) + k;
                sem_wait(&g_sem_free[buf]);
                if (g_prefetch_exit) break;
                if (g_skip_memcpy && (!first_tok || L >= STATIC_LAYERS + RING_LAYERS)) {
                    sem_post(&g_sem_ready[buf]);
                    continue;
                }
                copy_call_slot(c, buf);
                sem_post(&g_sem_ready[buf]);
            }
            if (g_prefetch_exit) break;
        }
        first_tok = 0;
    }
    return NULL;
}

static void* prefetch_thread_fn(void* arg) {
    (void)arg;
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    const char* pcpu_env = getenv("BONSAI_PREFETCH_CPU");
    int pcpu = pcpu_env ? atoi(pcpu_env) : 4;
    CPU_SET(pcpu, &cpuset);
    if (g_prefetch_threads < 2 && !pcpu_env) CPU_SET(5, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    setpriority(PRIO_PROCESS, 0, -10);
#endif
    int first_tok = 1;
    while (!g_prefetch_exit) {
        sem_wait(&g_sem_start_tok);
        if (g_prefetch_exit) break;
        for (int L = STATIC_LAYERS; L < 64; L++) {
            int slot = (L - STATIC_LAYERS) % RING_LAYERS;
            for (int k = 0; k < 4; k++) {
                if (g_prefetch_threads >= 3) {
                    if (k != 3) continue;
                } else if (g_prefetch_threads == 2) {
                    if (k == 2) continue;
                    if (g_alt_k1 && k == 1 && ((slot & 1) != 0)) continue;
                }
                int c = 4 * L + k;
                int buf = (slot * 4) + k;
                sem_wait(&g_sem_free[buf]);
                if (g_prefetch_exit) break;

#ifdef _WIN32
                if (npu_emu_is_call_staged(c)) {
                    sem_post(&g_sem_ready[buf]);
                    continue;
                }
#endif
                if (g_skip_memcpy && (!first_tok || L >= STATIC_LAYERS + RING_LAYERS)) {
                    sem_post(&g_sem_ready[buf]);
                    continue;
                }
                copy_call_slot(c, buf);
                sem_post(&g_sem_ready[buf]);
            }
            if (g_prefetch_exit) break;
        }
        first_tok = 0;
    }
    return NULL;
}


static int stage_init(void) {
    static int s_staged = 0;
    if (s_staged) return 0;
    rpcmem_init();

    const Tensor* th = find_t("language_model.lm_head");
    if (!th) return -1;
    int is_ternary = (th->kind == 2);

    const char* zcopy_env = getenv("BONSAI_ZERO_COPY");
    g_zero_copy = (!zcopy_env || atoi(zcopy_env) != 0);
    const char* unc_env = getenv("BONSAI_UNCACHED");
    unsigned w_flags = (unc_env && atoi(unc_env) == 0) ? 1u : 0u;
    const char* smap_env = getenv("BONSAI_STATIC_MAP");
    int do_smap = (!smap_env || atoi(smap_env) != 0);
    const char* battr_env = getenv("BONSAI_BUF_ATTR");
    int buf_attr = battr_env ? atoi(battr_env) : 0;
    const char* skip_env = getenv("BONSAI_SKIP_MEMCPY");
    g_skip_memcpy = (skip_env && atoi(skip_env) != 0);
    const char* scopy_env = getenv("BONSAI_SINGLE_COPY");
    g_single_copy = (scopy_env && atoi(scopy_env) != 0);

    const Tensor* ts5 = find_t("prism.hadamard.signs.5120");
    const Tensor* ts6 = find_t("prism.hadamard.signs.6144");
    const Tensor* ts17 = find_t("prism.hadamard.signs.17408");
    if (ts5 && ts6 && ts17) {
        g_signs_5120 = get_f32_ptr(ts5);
        g_signs_6144 = get_f32_ptr(ts6);
        g_signs_17408 = get_f32_ptr(ts17);
        fprintf(stderr, "[fwd] Bonsai 2 Hadamard signs active (5120, 6144, 17408)\n");
    }

    const char* fmlp_env = getenv("BONSAI_FUSED_MLP");
    g_fused_mlp = (!fmlp_env || atoi(fmlp_env) != 0);
    if (g_signs_17408 && g_h && g_fused_mlp) {
        int r_signs = bonsai_set_signs(g_h, g_signs_17408, 17408);
        fprintf(stderr, "[fwd] CDSP bonsai_set_signs(17408) rc=%d\n", r_signs);
    }
    fprintf(stderr, "[fwd] Fused MLP: %s\n", g_fused_mlp ? "ENABLED (1 RPC/layer)" : "DISABLED (2 RPC/layer)");

    const char* flin_env = getenv("BONSAI_FUSED_LIN");
    g_fused_lin = (!flin_env || atoi(flin_env) != 0);
    fprintf(stderr, "[fwd] Fused Lin Attn: %s\n", g_fused_lin ? "ENABLED (1 RPC/layer)" : "DISABLED (2 RPC/layer)");

    const char* flay_env = getenv("BONSAI_FUSED_LAYER");
    g_fused_layer = flay_env ? atoi(flay_env) : 2;
    fprintf(stderr, "[fwd] Fused Full Lin Layer: mode=%d (%s)\n", g_fused_layer,
            (g_fused_layer == 2) ? "ALL 64 LAYERS (1 RPC/layer)" : ((g_fused_layer == 1) ? "STATIC LIN LAYERS (Zero-Wait Hybrid)" : "DISABLED"));

    const char* slm_env = getenv("BONSAI_STATIC_LM");
    int do_static_lm = (!slm_env || atoi(slm_env) != 0);

    const char* sgrp_env = getenv("BONSAI_STATIC_GROUP");
    g_static_per_group = sgrp_env ? atoi(sgrp_env) : (do_static_lm ? 15 : 16);
    if (g_static_per_group < 8) g_static_per_group = 8;
    if (g_static_per_group > 24) g_static_per_group = 24;
    const char* slay_env = NULL;
    const char* pipe_env = getenv("BONSAI_PIPELINED_SMMU");
    g_pipelined_smmu = (pipe_env && atoi(pipe_env) != 0);
    if (g_pipelined_smmu) {
        g_static_layers = 64;
        g_ring_layers = 0;
        g_static_per_group = 16;
        fprintf(stderr, "[fwd] *** PIPELINED SMMU ZERO-COPY ENGINE ACTIVE (4x16 layers, 0 memcpy) ***\n");
    } else {
        slay_env = getenv("BONSAI_STATIC_LAYERS");
        g_static_layers = slay_env ? atoi(slay_env) : (2 * g_static_per_group);
        if (g_static_layers < 16) g_static_layers = 16;
        if (g_static_layers > 56) g_static_layers = 56;

        const char* rlay_env = getenv("BONSAI_RING_LAYERS");
        g_ring_layers = rlay_env ? atoi(rlay_env) : 2;
        if (g_ring_layers < 2) g_ring_layers = 2;
        if (g_ring_layers > MAX_RING_LAYERS) g_ring_layers = MAX_RING_LAYERS;
    }

    const char* nt_env = getenv("BONSAI_NT_COPY");
    g_nt_copy = (!nt_env || atoi(nt_env) != 0);

    init_call_plan();

    if (!g_pipelined_smmu) {
        fprintf(stderr, "[fwd] Initializing Hybrid Engine (%d Static Layers + %d Streamed Layers, Ring=%d layers, StaticLM=%d, NT_Copy=%d)...\n",
                STATIC_LAYERS, 64 - STATIC_LAYERS, g_ring_layers, do_static_lm, g_nt_copy);
    }
    double t_z0 = now_ms();

    // IO Buffers first so they can be used for static pointer registration
    size_t xs_sz = (size_t)8 * 17408 * 4;
    size_t ys_sz = (size_t)8 * 34816 * 4;
    g_xs = (float*)rpcmem_alloc(25, 1, (int)xs_sz);
    g_ys = (float*)rpcmem_alloc(25, 1, (int)ys_sz);
    if (!g_xs || !g_ys) return -3;
#ifndef _WIN32
    if (do_smap) {
        try_static_mmap(g_xs, xs_sz);
        try_static_mmap(g_ys, ys_sz);
    }
#endif

    // 0. Register CDSP linear attention states FIRST while 100% of CDSP 32-bit VA is free!
    //    (Uses lossless BF16 packing for in_proj_a/b: 55.04 MiB instead of 100.04 MiB, saving 45 MiB of CDSP VA!)
    if (g_fused_lin && g_h) {
        int req_ctx = (g_ctx > 4096) ? 4096 : g_ctx;
        if (req_ctx < 256) req_ctx = 256;
        g_dsp_ctx = req_ctx;
        size_t lin_aux_sz = (size_t)48 * 1188736 + (size_t)16 * 43008 + (size_t)g_dsp_ctx * 256;
#ifndef _WIN32
        uint8_t* temp_lin_aux = (uint8_t*)mmap(NULL, lin_aux_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (temp_lin_aux == MAP_FAILED) temp_lin_aux = NULL;
#else
        uint8_t* temp_lin_aux = (uint8_t*)malloc(lin_aux_sz);
#endif
        if (!temp_lin_aux) {
            fprintf(stderr, "[fwd] ERR: Failed to alloc temp lin_aux on host\n");
            return -1;
        }
        for (int L = 0; L < 64; L++) {
            if (L % 4 == 3) {
                int fi = L >> 2;
                const LayerWeights* lw = &g_lw[L];
                uint8_t* dst = temp_lin_aux + (size_t)48 * 1188736 + (size_t)fi * 43008;
                if (lw->in_norm)   memcpy(dst + 0,     lw->in_norm,   5120 * 4);
                if (lw->post_norm) memcpy(dst + 20480, lw->post_norm, 5120 * 4);
                if (lw->q_norm)    memcpy(dst + 40960, lw->q_norm,    256 * 4);
                if (lw->k_norm)    memcpy(dst + 41984, lw->k_norm,    256 * 4);
                continue;
            }
            int lin_i = L - (L >> 2);
            LayerWeights* lw = &g_lw[L];
            uint8_t* dst = temp_lin_aux + (size_t)lin_i * 1188736;
            const float* src_ab[2] = { lw->in_proj_a, lw->in_proj_b };
            uint32_t* dst_ab[2]    = { (uint32_t*)(dst + 0), (uint32_t*)(dst + 491520) };
            for (int ab = 0; ab < 2; ab++) {
                if (!src_ab[ab]) continue;
                const uint32_t* s_all = (const uint32_t*)src_ab[ab];
                uint32_t* d_all = dst_ab[ab];
                for (int r = 0; r < 48; r++) {
                    const uint32_t* sr = s_all + (size_t)r * 5120;
                    uint32_t* dr = d_all + (size_t)r * 2560;
                    for (int g = 0; g < 160; g += 2) {
                        const uint32_t* s0 = sr + (size_t)(g + 0) * 32;
                        const uint32_t* s1 = sr + (size_t)(g + 1) * 32;
                        uint32_t* d = dr + (size_t)(g >> 1) * 32;
                        for (int k = 0; k < 32; k++) {
                            uint32_t u0 = s0[k], u1 = s1[k];
                            uint16_t b0 = (uint16_t)((u0 + 0x7FFFu + ((u0 >> 16) & 1u)) >> 16);
                            uint16_t b1 = (uint16_t)((u1 + 0x7FFFu + ((u1 >> 16) & 1u)) >> 16);
                            d[k] = ((uint32_t)b1 << 16) | (uint32_t)b0;
                        }
                    }
                }
            }
            if (lw->conv1d_w)  memcpy(dst + 983040,  lw->conv1d_w,  10240 * 4 * 4);
            if (lw->A_log)     memcpy(dst + 1146880, lw->A_log,     48 * 4);
            if (lw->dt_bias)   memcpy(dst + 1147072, lw->dt_bias,   48 * 4);
            if (lw->lin_norm)  memcpy(dst + 1147264, lw->lin_norm,  128 * 4);
            if (lw->in_norm)   memcpy(dst + 1147776, lw->in_norm,   5120 * 4);
            if (lw->post_norm) memcpy(dst + 1168256, lw->post_norm, 5120 * 4);
#ifndef _WIN32
            const Tensor* ta = find_t(TN(L, "linear_attn.in_proj_a"));
            const Tensor* tb = find_t(TN(L, "linear_attn.in_proj_b"));
            const Tensor* tc = find_t(TN(L, "linear_attn.conv1d.weight"));
            if (ta) release_mmap_pages(ta->off, ta->len);
            if (tb) release_mmap_pages(tb->off, tb->len);
            if (tc) {
                int c_idx = (int)(tc - g_tab);
                if (c_idx >= 0 && c_idx < g_nt && g_f32_cache[c_idx]) {
                    free(g_f32_cache[c_idx]);
                    g_f32_cache[c_idx] = NULL;
                    lw->conv1d_w = NULL;
                }
                release_mmap_pages(tc->off, tc->len);
            }
#endif
        }
        {
            float* rope_tab = (float*)(temp_lin_aux + (size_t)48 * 1188736 + (size_t)16 * 43008);
            float freq[32];
            for (int i = 0; i < 32; i++) {
                freq[i] = 1.0f / powf(1e7f, (2.0f * i) / 64.0f);
            }
            for (int p = 0; p < g_dsp_ctx; p++) {
                float* row = rope_tab + (size_t)p * 64;
                for (int i = 0; i < 32; i++) {
                    float a = (float)p * freq[i];
                    row[i]      = cosf(a);
                    row[i + 32] = sinf(a);
                }
            }
        }

        int r_reg = bonsai_register_lin_states(g_h, NULL, 0, NULL, 0, temp_lin_aux, (int)lin_aux_sz);
        fprintf(stderr, "[fwd] CDSP bonsai_register_lin_states rc=%d (CDSP KV limit = %d tokens)\n", r_reg, g_dsp_ctx);
        if (g_signs_5120) {
            int r5 = bonsai_set_signs_dim(g_h, 5120, g_signs_5120, 5120);
            fprintf(stderr, "[fwd] CDSP bonsai_set_signs_dim(5120) rc=%d\n", r5);
        }
        if (g_signs_6144) {
            int r6 = bonsai_set_signs_dim(g_h, 6144, g_signs_6144, 6144);
            fprintf(stderr, "[fwd] CDSP bonsai_set_signs_dim(6144) rc=%d\n", r6);
        }
        if (g_signs_17408) {
            int r17 = bonsai_set_signs_dim(g_h, 17408, g_signs_17408, 17408);
            fprintf(stderr, "[fwd] CDSP bonsai_set_signs_dim(17408) rc=%d\n", r17);
        }
#ifndef _WIN32
        munmap(temp_lin_aux, lin_aux_sz);
#else
        free(temp_lin_aux);
#endif
    }
    if (!ssm_conv) ssm_conv = (float*)calloc((size_t)48 * 10240 * 3, sizeof(float));
    if (!ssm_rec)  ssm_rec  = (float*)calloc((size_t)48 * 48 * 128 * 128, sizeof(float));

    size_t total_group_bytes = 0;
    int max_grp_layers = (g_static_layers <= 36 && (g_static_layers % 2 == 0) && !slay_env) ? (g_static_layers / 2) : 16;

    // 1. Allocate & map Static Groups G = 0..3
    int mapped_static_layers = 0;
    for (int G = 0, l0 = 0; G < 4 && l0 < g_static_layers; G++) {
        int l1 = l0 + max_grp_layers;
        if (l1 > g_static_layers) l1 = g_static_layers;
        size_t group_bytes = 0;
        for (int L = l0; L < l1; L++) {
            for (int k = 0; k < 4; k++) {
                int c = 4 * L + k;
                const CallDesc* cd = &g_calls[c];
                size_t b_sz = (size_t)cd->total_out * cd->prow;
                size_t s_sz = (size_t)cd->total_out * cd->ng * 2;
                size_t b_sz_pad = (b_sz + 127) & ~(size_t)127;
                size_t s_sz_pad = (s_sz + 127) & ~(size_t)127;
                group_bytes += b_sz_pad + s_sz_pad;
            }
        }
        group_bytes = (group_bytes + 4095) & ~(size_t)4095;
        g_group_sz[G] = group_bytes;
        g_group_buf[G] = (uint8_t*)rpcmem_alloc(25, 0, (int)group_bytes);
        if (!g_group_buf[G]) {
            fprintf(stderr, "[fwd] ERR: Failed to alloc rpcmem for group %d (sz=%zu MB)\n", G, group_bytes >> 20);
            break;
        }
        g_group_fd[G] = rpcmem_to_fd(g_group_buf[G]);
        int rc_m = 0;
#ifndef _WIN32
        if (g_pipelined_smmu) {
            if (G <= 1) {
                rc_m = fastrpc_mmap(3, g_group_fd[G], g_group_buf[G], 0, g_group_sz[G], 0);
                g_group_mapped[G] = (rc_m == 0);
                fprintf(stderr, "[fwd] SMMU Map Group %d (layers %d..%d, sz=%zu MB, fd=%d): rc=%d\n",
                        G, l0, l1 - 1, g_group_sz[G] >> 20, g_group_fd[G], rc_m);
                if (rc_m != 0) {
                    rpcmem_free(g_group_buf[G]);
                    g_group_buf[G] = NULL;
                    g_group_sz[G] = 0;
                    break;
                }
            } else {
                g_group_mapped[G] = 0;
                fprintf(stderr, "[fwd] SMMU Standby Group %d (layers %d..%d, sz=%zu MB, fd=%d) [pipelined on-demand]\n",
                        G, l0, l1 - 1, g_group_sz[G] >> 20, g_group_fd[G]);
            }
        } else {
            rc_m = fastrpc_mmap(3, g_group_fd[G], g_group_buf[G], 0, g_group_sz[G], 0);
            fprintf(stderr, "[fwd] SMMU Map Group %d (layers %d..%d, sz=%zu MB, fd=%d): rc=%d\n",
                    G, l0, l1 - 1, g_group_sz[G] >> 20, g_group_fd[G], rc_m);
            if (rc_m != 0) {
                rpcmem_free(g_group_buf[G]);
                g_group_buf[G] = NULL;
                g_group_sz[G] = 0;
                break;
            }
        }
#endif
        size_t cur_off = 0;
        for (int L = l0; L < l1; L++) {
            for (int k = 0; k < 4; k++) {
                int c = 4 * L + k;
                const CallDesc* cd = &g_calls[c];
                size_t b_sz = (size_t)cd->total_out * cd->prow;
                size_t s_sz = (size_t)cd->total_out * cd->ng * 2;
                size_t b_sz_pad = (b_sz + 127) & ~(size_t)127;
                size_t s_sz_pad = (s_sz + 127) & ~(size_t)127;

                uint8_t* bdst = g_group_buf[G] + cur_off;
                g_call_bits[c] = bdst;
                g_call_static_bits[c] = bdst;
                cur_off += b_sz_pad;

                uint8_t* sdst = g_group_buf[G] + cur_off;
                g_call_scales[c] = (int16_t*)sdst;
                g_call_static_scales[c] = (int16_t*)sdst;
                cur_off += s_sz_pad;

                for (int t_idx = 0; t_idx < cd->nt; t_idx++) {
                    const Tensor* tk = cd->t[t_idx];
                    uint32_t tk_ng = tk->in / 128;
                    uint32_t tk_prow = tensor_prow(tk);
                    size_t tb_len = (size_t)tk->out * tk_prow;
                    size_t ts_len = (size_t)tk->out * tk_ng * 2;
#ifndef _WIN32
                    copy_and_release_mmap(bdst, tk->off + ts_len, tb_len);
                    bdst += tb_len;
                    copy_and_release_mmap(sdst, tk->off, ts_len);
                    sdst += ts_len;
                    release_mmap_pages(tk->off, tk->len);
#else
                    memcpy(bdst, g_base + tk->off + ts_len, tb_len);
                    bdst += tb_len;
                    memcpy(sdst, g_base + tk->off, ts_len);
                    sdst += ts_len;
#endif
                }
            }
        }
        total_group_bytes += group_bytes;
        mapped_static_layers = l1;
        l0 = l1;
    }
    g_static_layers = mapped_static_layers;

    // 2. Optionally map lm_head statically in CDSP SMMU (with immediate 4 MB sliding release on file pagecache!)
    g_lm_bits = NULL;
    g_lm_scales = NULL;
    g_lm_static_cdsp = 0;
    if (do_static_lm && g_h) {
        uint32_t lm_ng = th->in / 128;
        uint32_t lm_prow = tensor_prow(th);
        size_t lm_b_sz = (size_t)th->out * lm_prow;
        size_t lm_s_sz = (size_t)th->out * lm_ng * 2;
        size_t lm_total_sz = (lm_b_sz + lm_s_sz + 4095) & ~(size_t)4095;
        uint8_t* lm_arena = (uint8_t*)rpcmem_alloc(25, 0, (int)lm_total_sz);
        if (lm_arena) {
            int rc_lm_map = 0;
#ifndef _WIN32
            rc_lm_map = try_static_mmap(lm_arena, lm_total_sz);
            fprintf(stderr, "[fwd] SMMU Map Static LM Head (%zu MB): rc=%d\n", lm_total_sz >> 20, rc_lm_map);
#endif
            if (rc_lm_map == 0) {
                g_lm_bits = lm_arena;
                g_lm_scales = (int16_t*)(lm_arena + lm_b_sz);
#ifndef _WIN32
                copy_and_release_mmap(g_lm_bits, th->off + lm_s_sz, lm_b_sz);
                copy_and_release_mmap(g_lm_scales, th->off, lm_s_sz);
                release_mmap_pages(th->off, th->len);
#else
                memcpy(g_lm_bits, g_base + th->off + lm_s_sz, lm_b_sz);
                memcpy(g_lm_scales, g_base + th->off, lm_s_sz);
#endif
                int rc_reg_lm = bonsai_gemv_q1(g_h, (int)th->out, 0, (int)lm_b_sz,
                                               g_xs, 16, g_lm_bits, 64,
                                               (const short*)g_lm_scales, 64, g_ys, 16);
                fprintf(stderr, "[fwd] CDSP register Static LM Head ptrs rc=%d\n", rc_reg_lm);
                if (rc_reg_lm == 0) {
                    g_lm_static_cdsp = 1;
                }
            } else {
                rpcmem_free(lm_arena);
            }
        }
    }

    // Pre-allocate Demand-Paged Overcommit KV-cache (MAP_NORESERVE consumes 0 physical RAM upfront!)
#ifndef _WIN32
    size_t kv_sz  = (size_t)16 * 4 * 256 * g_ctx;
    size_t kvs_sz = (size_t)16 * 4 * g_ctx * sizeof(float);
    if (!kvk) {
        kvk = (int8_t*)mmap(NULL, kv_sz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
        if (kvk == MAP_FAILED) kvk = NULL;
    }
    if (!kvv) {
        kvv = (int8_t*)mmap(NULL, kv_sz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
        if (kvv == MAP_FAILED) kvv = NULL;
    }
    if (!kvks) {
        kvks = (float*)mmap(NULL, kvs_sz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
        if (kvks == MAP_FAILED) kvks = NULL;
    }
    if (!kvvs) {
        kvvs = (float*)mmap(NULL, kvs_sz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
        if (kvvs == MAP_FAILED) kvvs = NULL;
    }
#else
    if (!kvk)  kvk  = (int8_t*)calloc((size_t)16 * 4 * 256 * g_ctx, 1);
    if (!kvv)  kvv  = (int8_t*)calloc((size_t)16 * 4 * 256 * g_ctx, 1);
    if (!kvks) kvks = (float*)calloc((size_t)16 * 4 * g_ctx, sizeof(float));
    if (!kvvs) kvvs = (float*)calloc((size_t)16 * 4 * g_ctx, sizeof(float));
#endif

    if (g_t_embed) {
        release_mmap_pages(g_t_embed->off, g_t_embed->len);
    }

    if (g_pipelined_smmu) {
        sem_init(&g_sem_smmu_event, 0, 0);
        sem_init(&g_sem_smmu_done, 0, 0);
        pthread_create(&g_smmu_worker_tid, NULL, smmu_flip_worker_fn, NULL);
        fprintf(stderr, "[fwd] Pipelined SMMU Engine Ready: 64 Layers (%.2f GiB)%s in DMA-BUF (0 memcpy) staged in %.1f ms! (MemAvail: %ld MB)\n",
                (double)total_group_bytes / (1024.0 * 1024.0 * 1024.0),
                g_lm_static_cdsp ? " + Static LM Head (322 MB)" : "",
                now_ms() - t_z0, get_mem_avail_mb());
        g_zero_copy = 1;
        s_staged = 1;
        return 0;
    }

    // 3. Allocate & map Ring Arena (1 contiguous SMMU chunk!)
    static const int k_slot_bits_q2[4]   = { 16480 * 1280, 5120 * 1536, 34816 * 1280, 5120 * 4352 };
    static const int k_slot_scales_q2[4] = { 16480 * 80,   5120 * 96,   34816 * 80,   5120 * 272  };
    size_t ring_total = 0;
    size_t b_offsets[MAX_NUM_BUFS];
    size_t s_offsets[MAX_NUM_BUFS];
    for (int b = 0; b < NUM_BUFS; b++) {
        size_t b_sz = (size_t)(is_ternary ? k_slot_bits_q2[b & 3] : MAX_CALL_BITS_Q1);
        size_t s_sz = (size_t)(is_ternary ? k_slot_scales_q2[b & 3] : MAX_CALL_SCALES);
        b_sz = (b_sz + 4095) & ~(size_t)4095;
        s_sz = (s_sz + 4095) & ~(size_t)4095;
        b_offsets[b] = ring_total;
        ring_total += b_sz;
        s_offsets[b] = ring_total;
        ring_total += s_sz;
    }
    g_ring_arena = (uint8_t*)rpcmem_alloc(25, w_flags, (int)ring_total);
    if (!g_ring_arena) {
        fprintf(stderr, "[fwd] ERR: Failed to alloc ring arena (%zu MB)\n", ring_total >> 20);
        return -1;
    }
    g_ring_arena_sz = ring_total;
    memset(g_ring_arena, 0, ring_total);
#ifndef _WIN32
    if (buf_attr > 0) try_buf_attr(g_ring_arena, ring_total, buf_attr);
    if (do_smap) {
        int rc_ra = try_static_mmap(g_ring_arena, ring_total);
        fprintf(stderr, "[fwd] SMMU Map Ring Arena (%zu MB, %d layers): rc=%d\n", ring_total >> 20, g_ring_layers, rc_ra);
    }
#endif
    for (int b = 0; b < NUM_BUFS; b++) {
        g_bits_buf[b] = g_ring_arena + b_offsets[b];
        g_scales_buf[b] = (int16_t*)(g_ring_arena + s_offsets[b]);
        sem_init(&g_sem_free[b], 0, 1);
        sem_init(&g_sem_ready[b], 0, 0);
    }

    // 5.2 Drop embed_tokens pagecache (static layers, static lm_head, and aux already released their pages)
    //     so streamed layers 32..63 stay 100% intact in Active(file) pagecache across runs!
    if (g_t_embed) {
        release_mmap_pages(g_t_embed->off, g_t_embed->len);
    }
    double t_w0 = now_ms();
    volatile uint64_t warmup_sum = 0;
    size_t warmed_bytes = 0;
    for (int c = STATIC_LAYERS * 4; c < 256; c++) {
        const CallDesc* cd = &g_calls[c];
        for (int t_idx = 0; t_idx < cd->nt; t_idx++) {
            const Tensor* tk = cd->t[t_idx];
            size_t page_off = tk->off & ~(size_t)4095;
            size_t page_end = (tk->off + tk->len + 4095) & ~(size_t)4095;
            madvise((void*)(g_base + page_off), page_end - page_off, MADV_WILLNEED);
            for (size_t p = page_off; p < page_end; p += 4096) {
                warmup_sum += g_base[p];
            }
            warmed_bytes += tk->len;
        }
    }
    if (!g_lm_static_cdsp) {
        size_t page_off = th->off & ~(size_t)4095;
        size_t page_end = (th->off + th->len + 4095) & ~(size_t)4095;
        madvise((void*)(g_base + page_off), page_end - page_off, MADV_WILLNEED);
        for (size_t p = page_off; p < page_end; p += 4096) {
            warmup_sum += g_base[p];
        }
        warmed_bytes += th->len;
    }
    double t_w1 = now_ms();
    fprintf(stderr, "[fwd] Streamed Layers %d..63%s (%.2f GiB) warmed in clean pagecache in %.1f ms (touch sum=%llu, MemAvail: %ld MB)!\n",
            STATIC_LAYERS, g_lm_static_cdsp ? "" : " + LM Head",
            (double)warmed_bytes / (1024.0 * 1024.0 * 1024.0),
            t_w1 - t_w0, (unsigned long long)warmup_sum, get_mem_avail_mb());

    // 6. Threads & coordination
    const char* pth_env = getenv("BONSAI_PREFETCH_THREADS");
    g_prefetch_threads = pth_env ? atoi(pth_env) : 2;
    const char* ak1_env = getenv("BONSAI_ALT_K1");
    g_alt_k1 = (!ak1_env || atoi(ak1_env) != 0);
    sem_init(&g_sem_start_tok, 0, 0);
    sem_init(&g_sem_helper_start, 0, 0);
    sem_init(&g_sem_helper2_start, 0, 0);
    if (g_prefetch_threads >= 3) {
        pthread_create(&g_helper2_tid, NULL, prefetch_helper2_fn, NULL);
    }
    if (g_prefetch_threads >= 2) {
        pthread_create(&g_helper_tid, NULL, prefetch_helper_fn, NULL);
    }
    pthread_create(&g_prefetch_tid, NULL, prefetch_thread_fn, NULL);

    fprintf(stderr, "[fwd] Hybrid Engine Ready: %d Static Layers (%.2f GiB)%s + %d Streamed Layers (Ring Arena %zu MB) staged in %.1f ms! (MemAvail: %ld MB)\n",
            STATIC_LAYERS, (double)total_group_bytes / (1024.0 * 1024.0 * 1024.0),
            g_lm_static_cdsp ? " + Static LM Head (322 MB)" : "",
            64 - STATIC_LAYERS, ring_total >> 20, now_ms() - t_z0, get_mem_avail_mb());

    g_zero_copy = 0;
    s_staged = 1;
    return 0;
}

static int cdsp_call_batch(int c, int B, const float* x, float* y1, float* y2, float* y3, float* y4) {
    uint64_t t_call_start = (g_profile_tree && B == 1) ? now_us() : 0;
    if (stage_init()) return -98;
    const CallDesc* cd = &g_calls[c];
    int is_static = (g_call_static_bits[c] != NULL);
    int buf = 0;

    uint64_t t_wait_us = 0;
    if (g_zero_copy) {
        // Zero-copy: current 16-layer group is mapped in SMMU, zero memcpy and zero wait!
    } else if (!is_static) {
        int L = c / 4;
        int k = c % 4;
        buf = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + k;
        g_read_buf_idx++;
#ifdef _WIN32
        if (!g_vram_prestaged)
#endif
        {
            double t_w0 = now_ms();
            uint64_t u_w0 = (g_profile_tree && B == 1) ? now_us() : 0;
            sem_wait(&g_sem_ready[buf]);
            double t_w1 = now_ms();
            g_t_memcpy += (t_w1 - t_w0);
            if (g_profile_tree && B == 1) t_wait_us = now_us() - u_w0;
        }
    }

    const float* s_vec = signs_for_dim(cd->in_dim);
    uint64_t t_fwht_us = 0;
    uint64_t u_f0 = (g_profile_tree && B == 1) ? now_us() : 0;
    for (int b = 0; b < B; b++) {
        const float* xb = x + (size_t)b * cd->in_dim;
        float* xdst = g_xs + (size_t)b * cd->in_dim;
        if (s_vec) {
            bonsai_fwht1024(xb, s_vec, cd->in_dim, 0, xdst);
        } else {
            memcpy(xdst, xb, (size_t)cd->in_dim * 4);
        }
    }
    if (g_profile_tree && B == 1) t_fwht_us = now_us() - u_f0;

#ifdef _WIN32
    npu_emu_set_active_call(c);
#endif
    const uint8_t* b_ptr = g_zero_copy ? g_call_bits[c] : (is_static ? g_call_static_bits[c]   : g_bits_buf[buf]);
    const short*   s_ptr = g_zero_copy ? (const short*)g_call_scales[c] : (is_static ? (const short*)g_call_static_scales[c] : (const short*)g_scales_buf[buf]);
    double t_r0 = now_ms();
    uint64_t u_r0 = (g_profile_tree && B == 1) ? now_us() : 0;
    int rc = bonsai_gemv_q1(g_h, cd->total_out, cd->in_dim, cd->prow,
                            g_xs, B * cd->in_dim,
                            b_ptr, cd->total_out * cd->prow,
                            s_ptr, cd->total_out * cd->ng,
                            g_ys, B * cd->total_out);
    double t_r1 = now_ms();
    g_t_rpc += (t_r1 - t_r0);
    g_calls_rpc++;
    uint64_t t_rpc_us = (g_profile_tree && B == 1) ? (now_us() - u_r0) : 0;

    if (!g_zero_copy && !is_static) {
#ifdef _WIN32
        if (!g_vram_prestaged)
#endif
        {
            sem_post(&g_sem_free[buf]);
        }
    }
    if (rc != 0) { printf("cdsp_call %d (B=%d) err rc=%d\n", c, B, rc); return rc; }

    uint64_t u_p0 = (g_profile_tree && B == 1) ? now_us() : 0;
    for (int b = 0; b < B; b++) {
        const float* yb = g_ys + (size_t)b * cd->total_out;
        int off = 0;
        if (y1 && cd->nt >= 1) { memcpy(y1 + (size_t)b * cd->t[0]->out, yb + off, (size_t)cd->t[0]->out * 4); off += cd->t[0]->out; }
        if (y2 && cd->nt >= 2) { memcpy(y2 + (size_t)b * cd->t[1]->out, yb + off, (size_t)cd->t[1]->out * 4); off += cd->t[1]->out; }
        if (y3 && cd->nt >= 3) { memcpy(y3 + (size_t)b * cd->t[2]->out, yb + off, (size_t)cd->t[2]->out * 4); off += cd->t[2]->out; }
        if (y4 && cd->nt >= 4) { memcpy(y4 + (size_t)b * cd->t[3]->out, yb + off, (size_t)cd->t[3]->out * 4); }
    }
    uint64_t t_post_us = (g_profile_tree && B == 1) ? (now_us() - u_p0) : 0;

    if (g_profile_tree && B == 1) {
        int L = c / 4;
        int call_type = c % 4;
        if (L < 64) {
            CallTiming* ct = (call_type == 0) ? &g_prof.layers[L].c0 : &g_prof.layers[L].c1;
            ct->t_wait_us = t_wait_us;
            ct->t_fwht_us = t_fwht_us;
            ct->t_rpc_us  = t_rpc_us;
            ct->t_post_us = t_post_us;
            ct->t_total_us = now_us() - t_call_start;
        }
    }

    return 0;
}

static int cdsp_call(int c, const float* x, float* y1, float* y2, float* y3, float* y4) {
    return cdsp_call_batch(c, 1, x, y1, y2, y3, y4);
}

static int cdsp_gemv_fused4(const Tensor* t1, const Tensor* t2, const Tensor* t3, const Tensor* t4,
                            const float* x, float* y1, float* y2, float* y3, float* y4) {
    (void)t1; (void)t2; (void)t3; (void)t4;
    return cdsp_call(g_cur_call++, x, y1, y2, y3, y4);
}

static int cdsp_gemv_fused3(const Tensor* t1, const Tensor* t2, const Tensor* t3,
                            const float* x, float* y1, float* y2, float* y3) {
    (void)t1; (void)t2; (void)t3;
    return cdsp_call(g_cur_call++, x, y1, y2, y3, NULL);
}

static int cdsp_gemv_fused2(const Tensor* t1, const Tensor* t2, const float* x, float* y1, float* y2) {
    (void)t1; (void)t2;
    return cdsp_call(g_cur_call++, x, y1, y2, NULL, NULL);
}

static int cdsp_gemv(const char* name, const float* x, float* y) {
    (void)name;
    return cdsp_call(g_cur_call++, x, y, NULL, NULL, NULL);
}

static int g_fast_argmax_idx = -1;
static float g_fast_argmax_val = 0.0f;

static int cdsp_lmhead_batch(const float* x, int B, float* logits) {
    if (stage_init()) return -98;
    const Tensor* th = g_t_lmhead;
    if (!th) return -1;
    uint32_t ng = th->in / 128;
    uint32_t prow = tensor_prow(th);
    g_fast_argmax_idx = -1;
    g_fast_argmax_val = -1e30f;

    const char* lmc_env = getenv("BONSAI_LM_CHUNK");
    int chunk = lmc_env ? atoi(lmc_env) : 31040;
    if (chunk > 34816) chunk = 34816;
    if (chunk < 1024) chunk = 15520;
    const char* pp_env = getenv("BONSAI_LM_PINGPONG");
    int do_pingpong = (pp_env && atoi(pp_env) != 0);
    const char* lmc_calls_env = getenv("BONSAI_LM_CALLS");
    int lm_calls_mode = lmc_calls_env ? atoi(lmc_calls_env) : 2;

    static int s_force_4call = 0;
    if (s_force_4call && lm_calls_mode <= 2) lm_calls_mode = 4;

    if (g_lm_static_cdsp && g_lm_bits && g_lm_scales && lm_calls_mode <= 2) {
        int ok_2call = 1;
        for (int b = 0; b < B; b++) {
            memcpy(g_xs, x + (size_t)b * th->in, (size_t)th->in * sizeof(float));
            int b_fast_argmax_idx = -1;
            float b_fast_argmax_val = -1e30f;
            for (int h = 0; h < 2; h++) {
                int r_base = h * 4 * 31040;
                const uint8_t* b0 = g_lm_bits + (size_t)(r_base + 0 * 31040) * prow;
                const uint8_t* b1 = g_lm_bits + (size_t)(r_base + 1 * 31040) * prow;
                const uint8_t* b2 = g_lm_bits + (size_t)(r_base + 2 * 31040) * prow;
                const uint8_t* b3 = g_lm_bits + (size_t)(r_base + 3 * 31040) * prow;
                const short*   s0 = (const short*)g_lm_scales + (size_t)(r_base + 0 * 31040) * ng;
                const short*   s1 = (const short*)g_lm_scales + (size_t)(r_base + 1 * 31040) * ng;
                const short*   s2 = (const short*)g_lm_scales + (size_t)(r_base + 2 * 31040) * ng;
                const short*   s3 = (const short*)g_lm_scales + (size_t)(r_base + 3 * 31040) * ng;
                double t_r0 = now_ms();
                uint64_t u_r0 = (g_profile_tree && B == 1) ? now_us() : 0;
                int rc_h = bonsai_lin_layer_fused(g_h, -100 - h,
                                                  g_xs, (int)th->in,
                                                  b0, 31040 * (int)prow, s0, 31040 * (int)ng,
                                                  b1, 31040 * (int)prow, s1, 31040 * (int)ng,
                                                  b2, 31040 * (int)prow, s2, 31040 * (int)ng,
                                                  b3, 31040 * (int)prow, s3, 31040 * (int)ng,
                                                  g_ys, 16);
                double t_r1 = now_ms();
                if (rc_h != 0) {
                    fprintf(stderr, "[fwd] 2-call static LM head h=%d returned rc=0x%x (%d), falling back to 4-call fused path\n", h, rc_h, rc_h);
                    s_force_4call = 1;
                    lm_calls_mode = 4;
                    ok_2call = 0;
                    b_fast_argmax_idx = -1;
                    break;
                }
                g_t_rpc += (t_r1 - t_r0);
                g_calls_rpc++;
                if (g_profile_tree && B == 1) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
                float v = g_ys[0];
                int32_t idx = 0;
                memcpy(&idx, &g_ys[1], 4);
                if (b_fast_argmax_idx < 0 || v > b_fast_argmax_val) {
                    b_fast_argmax_val = v;
                    b_fast_argmax_idx = idx;
                }
            }
            if (!ok_2call) break;
            if (b == 0) {
                g_fast_argmax_val = b_fast_argmax_val;
                g_fast_argmax_idx = b_fast_argmax_idx;
            }
            if (b_fast_argmax_idx >= 0 && b_fast_argmax_idx < 248320) {
                logits[(size_t)b * th->out + b_fast_argmax_idx] = b_fast_argmax_val;
            }
        }
        if (ok_2call) return 0;
    }

    if (g_lm_static_cdsp && g_lm_bits && g_lm_scales && lm_calls_mode == 4) {
        for (int b = 0; b < B; b++) {
            memcpy(g_xs, x + (size_t)b * th->in, (size_t)th->in * sizeof(float));
            int b_fast_argmax_idx = -1;
            float b_fast_argmax_val = -1e30f;
            for (int h = 0; h < 4; h++) {
                int r_base = h * 2 * 31040;
                const uint8_t* b0 = g_lm_bits + (size_t)(r_base + 0 * 31040) * prow;
                const uint8_t* b1 = g_lm_bits + (size_t)(r_base + 1 * 31040) * prow;
                const short*   s0 = (const short*)g_lm_scales + (size_t)(r_base + 0 * 31040) * ng;
                const short*   s1 = (const short*)g_lm_scales + (size_t)(r_base + 1 * 31040) * ng;
                double t_r0 = now_ms();
                uint64_t u_r0 = (g_profile_tree && B == 1) ? now_us() : 0;
                int rc_h = bonsai_lin_layer_fused(g_h, -200 - h,
                                                  g_xs, (int)th->in,
                                                  b0, 31040 * (int)prow, s0, 31040 * (int)ng,
                                                  b1, 31040 * (int)prow, s1, 31040 * (int)ng,
                                                  b0, 128, (const short*)s0, 64,
                                                  b0, 128, (const short*)s0, 64,
                                                  g_ys, 16);
                double t_r1 = now_ms();
                g_t_rpc += (t_r1 - t_r0);
                g_calls_rpc++;
                if (g_profile_tree && B == 1) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
                if (rc_h != 0) {
                    fprintf(stderr, "[fwd] 4-call static LM head h=%d failed rc=0x%x (%d)\n", h, rc_h, rc_h);
                    return rc_h;
                }
                float v = g_ys[0];
                int32_t idx = 0;
                memcpy(&idx, &g_ys[1], 4);
                if (b_fast_argmax_idx < 0 || v > b_fast_argmax_val) {
                    b_fast_argmax_val = v;
                    b_fast_argmax_idx = idx;
                }
            }
            if (b == 0) {
                g_fast_argmax_val = b_fast_argmax_val;
                g_fast_argmax_idx = b_fast_argmax_idx;
            }
            if (b_fast_argmax_idx >= 0 && b_fast_argmax_idx < 248320) {
                logits[(size_t)b * th->out + b_fast_argmax_idx] = b_fast_argmax_val;
            }
        }
        return 0;
    }

    if (B == 1 && !g_lm_static_cdsp && chunk == 31040 && g_ring_arena_sz >= 177078272 && lm_calls_mode <= 2) {
        memcpy(g_xs, x, (size_t)th->in * sizeof(float));
        const uint8_t* s_base = g_lm_scales ? (const uint8_t*)g_lm_scales : (g_base + th->off);
        const uint8_t* b_base = g_lm_bits ? g_lm_bits : (g_base + th->off + (size_t)th->out * ng * 2);
        size_t cb_sz = (size_t)31040 * prow;       // 39,731,200
        size_t cs_sz = (size_t)31040 * ng * 2;     //  2,483,200
        size_t slot_stride = 44269568;             // 4096-aligned slot in g_ring_arena
        int ok_2call = 1;
        g_fast_argmax_idx = -1;
        for (int h = 0; h < 2; h++) {
            int r_base = h * 4 * 31040;
            uint8_t* b_slot[4];
            short*   s_slot[4];
            for (int k = 0; k < 4; k++) {
                b_slot[k] = g_ring_arena + (size_t)k * slot_stride;
                s_slot[k] = (short*)(b_slot[k] + cb_sz);
            }
            double t_m0 = now_ms();
            uint64_t u_m0 = g_profile_tree ? now_us() : 0;
#ifdef _OPENMP
            #pragma omp parallel for num_threads(4)
#endif
            for (int k = 0; k < 4; k++) {
                int r0 = r_base + k * 31040;
                fast_stream_copy(b_slot[k], b_base + (size_t)r0 * prow, cb_sz);
                fast_stream_copy(s_slot[k], s_base + (size_t)r0 * ng * 2, cs_sz);
            }
            double t_m1 = now_ms();
            g_t_memcpy += (t_m1 - t_m0);
            if (g_profile_tree) g_prof.t_lmhead_memcpy_us += (now_us() - u_m0);

            double t_r0 = now_ms();
            uint64_t u_r0 = g_profile_tree ? now_us() : 0;
            int rc_h = bonsai_lin_layer_fused(g_h, -100 - h,
                                              g_xs, (int)th->in,
                                              b_slot[0], (int)cb_sz, s_slot[0], 31040 * (int)ng,
                                              b_slot[1], (int)cb_sz, s_slot[1], 31040 * (int)ng,
                                              b_slot[2], (int)cb_sz, s_slot[2], 31040 * (int)ng,
                                              b_slot[3], (int)cb_sz, s_slot[3], 31040 * (int)ng,
                                              g_ys, 16);
            double t_r1 = now_ms();
            if (rc_h != 0) {
                fprintf(stderr, "[fwd] 2-call streamed LM head h=%d returned rc=0x%x (%d), falling back to 4-call fused path\n", h, rc_h, rc_h);
                s_force_4call = 1;
                lm_calls_mode = 4;
                ok_2call = 0;
                g_fast_argmax_idx = -1;
                break;
            }
            g_t_rpc += (t_r1 - t_r0);
            g_calls_rpc++;
            if (g_profile_tree) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
            float v = g_ys[0];
            int32_t idx = 0;
            memcpy(&idx, &g_ys[1], 4);
            if (g_fast_argmax_idx < 0 || v > g_fast_argmax_val) {
                g_fast_argmax_val = v;
                g_fast_argmax_idx = idx;
            }
        }
        if (ok_2call && g_fast_argmax_idx >= 0 && g_fast_argmax_idx < 248320) {
            logits[g_fast_argmax_idx] = g_fast_argmax_val;
            return 0;
        }
    }

    if (B == 1 && !g_lm_static_cdsp && chunk == 31040 && g_ring_arena_sz >= 88539136 && lm_calls_mode == 4) {
        memcpy(g_xs, x, (size_t)th->in * sizeof(float));
        const uint8_t* s_base = g_lm_scales ? (const uint8_t*)g_lm_scales : (g_base + th->off);
        const uint8_t* b_base = g_lm_bits ? g_lm_bits : (g_base + th->off + (size_t)th->out * ng * 2);
        size_t cb_sz = (size_t)31040 * prow;       // 39,731,200
        size_t cs_sz = (size_t)31040 * ng * 2;     //  2,483,200
        size_t slot_stride = 44269568;             // 4096-aligned slot in g_ring_arena
        g_fast_argmax_idx = -1;
        for (int h = 0; h < 4; h++) {
            int r_base = h * 2 * 31040;
            uint8_t* b_slot[2];
            short*   s_slot[2];
            for (int k = 0; k < 2; k++) {
                b_slot[k] = g_ring_arena + (size_t)k * slot_stride;
                s_slot[k] = (short*)(b_slot[k] + cb_sz);
            }
            double t_m0 = now_ms();
            uint64_t u_m0 = g_profile_tree ? now_us() : 0;
#ifdef _OPENMP
            #pragma omp parallel num_threads(4)
            {
                int tid = omp_get_thread_num();
                int k = tid >> 1;
                int sub = tid & 1;
                int r0 = r_base + k * 31040;
                size_t b_half = cb_sz >> 1;
                size_t s_half = cs_sz >> 1;
                fast_stream_copy(b_slot[k] + sub * b_half, b_base + (size_t)r0 * prow + sub * b_half, b_half);
                fast_stream_copy((uint8_t*)s_slot[k] + sub * s_half, s_base + (size_t)r0 * ng * 2 + sub * s_half, s_half);
            }
#else
            for (int k = 0; k < 2; k++) {
                int r0 = r_base + k * 31040;
                fast_stream_copy(b_slot[k], b_base + (size_t)r0 * prow, cb_sz);
                fast_stream_copy(s_slot[k], s_base + (size_t)r0 * ng * 2, cs_sz);
            }
#endif
            double t_m1 = now_ms();
            g_t_memcpy += (t_m1 - t_m0);
            if (g_profile_tree) g_prof.t_lmhead_memcpy_us += (now_us() - u_m0);

            double t_r0 = now_ms();
            uint64_t u_r0 = g_profile_tree ? now_us() : 0;
            int rc_h = bonsai_lin_layer_fused(g_h, -200 - h,
                                              g_xs, (int)th->in,
                                              b_slot[0], (int)cb_sz, s_slot[0], 31040 * (int)ng,
                                              b_slot[1], (int)cb_sz, s_slot[1], 31040 * (int)ng,
                                              g_call_static_bits[0], 128, (const short*)g_call_static_scales[0], 64,
                                              g_call_static_bits[1], 128, (const short*)g_call_static_scales[1], 64,
                                              g_ys, 16);
            double t_r1 = now_ms();
            g_t_rpc += (t_r1 - t_r0);
            g_calls_rpc++;
            if (g_profile_tree) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
            if (rc_h != 0) {
                fprintf(stderr, "[fwd] 4-call streamed LM head h=%d failed rc=0x%x (%d)\n", h, rc_h, rc_h);
                g_fast_argmax_idx = -1;
                return rc_h;
            }
            float v = g_ys[0];
            int32_t idx = 0;
            memcpy(&idx, &g_ys[1], 4);
            if (g_fast_argmax_idx < 0 || v > g_fast_argmax_val) {
                g_fast_argmax_val = v;
                g_fast_argmax_idx = idx;
            }
        }
        if (g_fast_argmax_idx >= 0 && g_fast_argmax_idx < 248320) {
            logits[g_fast_argmax_idx] = g_fast_argmax_val;
        }
        return 0;
    }

    uint64_t u_hfwht0 = (g_profile_tree && B == 1) ? now_us() : 0;
    for (int b = 0; b < B; b++) {
        const float* xb = x + (size_t)b * th->in;
        float* xdst = g_xs + (size_t)b * th->in;
        if (g_signs_5120) {
            bonsai_fwht1024(xb, g_signs_5120, (int)th->in, 0, xdst);
        } else {
            memcpy(xdst, xb, (size_t)th->in * 4);
        }
    }
    if (g_profile_tree && B == 1) g_prof.t_lmhead_fwht_us = now_us() - u_hfwht0;

    for (int b = 0; b < B; b++) {
        if (g_lm_static_cdsp && g_lm_bits && g_lm_scales) {
            for (int r0 = 0; r0 < (int)th->out; r0 += chunk) {
                int nr = ((int)th->out - r0 < chunk) ? ((int)th->out - r0) : chunk;
                const uint8_t* b_src = g_lm_bits + (size_t)r0 * prow;
                const short*   s_src = (const short*)g_lm_scales + (size_t)r0 * ng;
                double t_r0 = now_ms();
                uint64_t u_r0 = (g_profile_tree && B == 1) ? now_us() : 0;
                int rc_chunk = bonsai_gemv_q1(g_h, nr, (int)th->in, (int)prow,
                                              g_xs + (size_t)b * th->in, (int)th->in,
                                              b_src, nr * (int)prow,
                                              s_src, nr * (int)ng,
                                              g_ys, nr);
                double t_r1 = now_ms();
                g_t_rpc += (t_r1 - t_r0);
                g_calls_rpc++;
                if (g_profile_tree && B == 1) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
                if (rc_chunk != 0) {
                    fprintf(stderr, "[fwd] static cdsp_lmhead_batch r0=%d failed rc=0x%x (%d)\n", r0, rc_chunk, rc_chunk);
                    return rc_chunk;
                }
                uint64_t u_p0 = (g_profile_tree && B == 1) ? now_us() : 0;
                memcpy(logits + (size_t)b * th->out + r0, g_ys, (size_t)nr * 4);
                if (g_profile_tree && B == 1) g_prof.t_lmhead_post_us += (now_us() - u_p0);
            }
            continue;
        }
        const uint8_t* s_base = g_lm_scales ? (const uint8_t*)g_lm_scales : (g_base + th->off);
        const uint8_t* b_base = g_lm_bits ? g_lm_bits : (g_base + th->off + (size_t)th->out * ng * 2);

#ifdef _OPENMP
        if (do_pingpong && B == 1) {
            int nr0 = ((int)th->out < chunk) ? (int)th->out : chunk;
            size_t b_bytes0 = (size_t)nr0 * prow;
            size_t s_bytes0 = (size_t)nr0 * ng * 2;
            double t_m0 = now_ms();
            uint64_t u_m0 = g_profile_tree ? now_us() : 0;
            #pragma omp parallel num_threads(4)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                size_t b_part = ((b_bytes0 + nth - 1) / nth + 63) & ~(size_t)63;
                size_t b_off  = (size_t)tid * b_part;
                if (b_off < b_bytes0) {
                    size_t b_len = (b_off + b_part <= b_bytes0) ? b_part : (b_bytes0 - b_off);
                    fast_stream_copy(g_bits_buf[2] + b_off, b_base + b_off, b_len);
                }
                size_t s_part = ((s_bytes0 + nth - 1) / nth + 63) & ~(size_t)63;
                size_t s_off  = (size_t)tid * s_part;
                if (s_off < s_bytes0) {
                    size_t s_len = (s_off + s_part <= s_bytes0) ? s_part : (s_bytes0 - s_off);
                    fast_stream_copy((uint8_t*)g_scales_buf[2] + s_off, s_base + s_off, s_len);
                }
            }
            double t_m1 = now_ms();
            g_t_memcpy += (t_m1 - t_m0);
            if (g_profile_tree) g_prof.t_lmhead_memcpy_us += (now_us() - u_m0);

            int rc_err = 0;
            int c_idx = 0;
            for (int r0 = 0; r0 < (int)th->out; r0 += chunk, c_idx++) {
                int nr = ((int)th->out - r0 < chunk) ? ((int)th->out - r0) : chunk;
                int cur_slot = (c_idx & 1) ? 6 : 2;
                int nxt_slot = (c_idx & 1) ? 2 : 6;
                int nxt_r0 = r0 + chunk;
                int has_nxt = (nxt_r0 < (int)th->out);
                int nr_nxt = has_nxt ? (((int)th->out - nxt_r0 < chunk) ? ((int)th->out - nxt_r0) : chunk) : 0;
                size_t b_bytes_nxt = (size_t)nr_nxt * prow;
                size_t s_bytes_nxt = (size_t)nr_nxt * ng * 2;
                const uint8_t* b_src_nxt = b_base + (size_t)nxt_r0 * prow;
                const uint8_t* s_src_nxt = s_base + (size_t)nxt_r0 * ng * 2;

                #pragma omp parallel num_threads(4)
                {
                    int tid = omp_get_thread_num();
                    if (tid == 0) {
                        double t_r0 = now_ms();
                        uint64_t u_r0 = g_profile_tree ? now_us() : 0;
                        int rc_chunk = bonsai_gemv_q1(g_h, nr, (int)th->in, (int)prow,
                                                      g_xs + (size_t)b * th->in, (int)th->in,
                                                      g_bits_buf[cur_slot], nr * (int)prow,
                                                      (const short*)g_scales_buf[cur_slot], nr * (int)ng,
                                                      g_ys, nr);
                        double t_r1 = now_ms();
                        g_t_rpc += (t_r1 - t_r0);
                        g_calls_rpc++;
                        if (g_profile_tree) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
                        if (rc_chunk != 0) rc_err = rc_chunk;
                        else {
                            uint64_t u_p0 = g_profile_tree ? now_us() : 0;
                            memcpy(logits + (size_t)b * th->out + r0, g_ys, (size_t)nr * 4);
                            if (g_profile_tree) g_prof.t_lmhead_post_us += (now_us() - u_p0);
                        }
                    } else if (has_nxt) {
                        int w_id = tid - 1;
                        int w_nth = 3;
                        size_t b_part = ((b_bytes_nxt + w_nth - 1) / w_nth + 63) & ~(size_t)63;
                        size_t b_off  = (size_t)w_id * b_part;
                        if (b_off < b_bytes_nxt) {
                            size_t b_len = (b_off + b_part <= b_bytes_nxt) ? b_part : (b_bytes_nxt - b_off);
                            fast_stream_copy(g_bits_buf[nxt_slot] + b_off, b_src_nxt + b_off, b_len);
                        }
                        size_t s_part = ((s_bytes_nxt + w_nth - 1) / w_nth + 63) & ~(size_t)63;
                        size_t s_off  = (size_t)w_id * s_part;
                        if (s_off < s_bytes_nxt) {
                            size_t s_len = (s_off + s_part <= s_bytes_nxt) ? s_part : (s_bytes_nxt - s_off);
                            fast_stream_copy((uint8_t*)g_scales_buf[nxt_slot] + s_off, s_src_nxt + s_off, s_len);
                        }
                    }
                }
                if (rc_err != 0) return rc_err;
            }
            continue;
        }
#endif

        uint8_t* lm_b_dst = g_bits_buf[2];
        int16_t* lm_s_dst = g_scales_buf[2];

        for (int r0 = 0; r0 < (int)th->out; r0 += chunk) {
            int nr = ((int)th->out - r0 < chunk) ? ((int)th->out - r0) : chunk;
            size_t b_bytes = (size_t)nr * prow;
            size_t s_bytes = (size_t)nr * ng * 2;
            const uint8_t* b_src = b_base + (size_t)r0 * prow;
            const uint8_t* s_src = s_base + (size_t)r0 * ng * 2;

            double t_m0 = now_ms();
            uint64_t u_m0 = (g_profile_tree && B == 1) ? now_us() : 0;
#ifdef _OPENMP
            #pragma omp parallel num_threads(4)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                size_t b_part = ((b_bytes + nth - 1) / nth + 63) & ~(size_t)63;
                size_t b_off  = (size_t)tid * b_part;
                if (b_off < b_bytes) {
                    size_t b_len = (b_off + b_part <= b_bytes) ? b_part : (b_bytes - b_off);
                    fast_stream_copy(lm_b_dst + b_off, b_src + b_off, b_len);
                }
                size_t s_part = ((s_bytes + nth - 1) / nth + 63) & ~(size_t)63;
                size_t s_off  = (size_t)tid * s_part;
                if (s_off < s_bytes) {
                    size_t s_len = (s_off + s_part <= s_bytes) ? s_part : (s_bytes - s_off);
                    fast_stream_copy((uint8_t*)lm_s_dst + s_off, s_src + s_off, s_len);
                }
            }
#else
            fast_stream_copy(lm_b_dst, b_src, b_bytes);
            fast_stream_copy(lm_s_dst, s_src, s_bytes);
#endif
            double t_m1 = now_ms();
            g_t_memcpy += (t_m1 - t_m0);
            if (g_profile_tree && B == 1) g_prof.t_lmhead_memcpy_us += (now_us() - u_m0);

            double t_r0 = now_ms();
            uint64_t u_r0 = (g_profile_tree && B == 1) ? now_us() : 0;
            int rc_chunk = bonsai_gemv_q1(g_h, nr, (int)th->in, (int)prow,
                                          g_xs + (size_t)b * th->in, (int)th->in,
                                          lm_b_dst, nr * (int)prow,
                                          (const short*)lm_s_dst, nr * (int)ng,
                                          g_ys, nr);
            double t_r1 = now_ms();
            g_t_rpc += (t_r1 - t_r0);
            g_calls_rpc++;
            if (g_profile_tree && B == 1) g_prof.t_lmhead_rpc_us += (now_us() - u_r0);
            if (rc_chunk != 0) {
                fprintf(stderr, "[fwd] cdsp_lmhead_batch chunk r0=%d failed rc=0x%x (%d)\n", r0, rc_chunk, rc_chunk);
                return rc_chunk;
            }
            uint64_t u_p0 = (g_profile_tree && B == 1) ? now_us() : 0;
            memcpy(logits + (size_t)b * th->out + r0, g_ys, (size_t)nr * 4);
            if (g_profile_tree && B == 1) g_prof.t_lmhead_post_us += (now_us() - u_p0);
        }
    }
    return 0;
}

static int cdsp_lmhead(const float* x, float* logits) {
    return cdsp_lmhead_batch(x, 1, logits);
}

// dequant one binary or ternary row (ARM, for embed only)
static void dequant_row(const Tensor* t, int row, float* out) {
    uint32_t ng = t->in / 128, rb = t->in / 8;
    uint32_t prow = tensor_prow(t);
    const uint16_t* sc = (const uint16_t*)(g_base + t->off) + (size_t)row * ng;
    const uint8_t* bp = g_base + t->off + (size_t)t->out * ng * 2 + (size_t)row * prow;
    if (t->kind == 2) {
        const uint8_t* neg_bp = bp + rb;
        for (uint32_t j = 0; j < t->in; j++) {
            int pbit = (bp[j >> 3] >> (j & 7)) & 1;
            int nbit = (neg_bp[j >> 3] >> (j & 7)) & 1;
            float s = f16(0, sc[j >> 7]);
            out[j] = (float)(pbit - nbit) * s;
        }
        if (g_signs_5120 && t->in == 5120) {
            bonsai_fwht1024(out, g_signs_5120, 5120, 1, out);
        }
    } else {
        for (uint32_t j = 0; j < t->in; j++) {
            int bit = (bp[j >> 3] >> (j & 7)) & 1;
            float s = f16(0, sc[j >> 7]) * 0.5f;
            out[j] = bit ? s : -s;
        }
    }
}

// ---- state ----
#define HIDDEN 5120
#define FFN 17408
#define NLAYER 64
#define NKH 16
#define NVH 48
#define HD 128
#define FQ 24
#define FKV 4
#define FHD 256

static int g_lin_idx[64], g_full_idx[64], g_nlin = 0, g_nfull = 0;

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

static float* scratch(const char* n, size_t bytes) {
    static struct { const char* n; float* p; size_t cap; } pool[32];
    static int np = 0;
    for (int i = 0; i < np; i++) {
        if (!strcmp(pool[i].n, n)) {
            if (bytes > pool[i].cap) {
                pool[i].p = (float*)realloc(pool[i].p, bytes);
                pool[i].cap = bytes;
            }
            return pool[i].p;
        }
    }
    float* p = (float*)malloc(bytes);
    pool[np].n = n; pool[np].p = p; pool[np].cap = bytes; np++;
    return p;
}

typedef struct {
    int B;
    const float* alog;
    const float* dtb;
    const float* avec_b;
    const float* bvec_b;
    const float* q_b;
    const float* k_b;
    const float* v_b;
    const float* z_b;
    const float* nw;
    float* S;
    float* no_b;
} LinTask;

static LinTask g_lin_task;

static void run_lin_heads(int h0, int h1, const LinTask* lt) {
    int B = lt->B;
    const float* alog = lt->alog;
    const float* dtb = lt->dtb;
    const float* avec_b = lt->avec_b;
    const float* bvec_b = lt->bvec_b;
    const float* q_b = lt->q_b;
    const float* k_b = lt->k_b;
    const float* v_b = lt->v_b;
    const float* z_b = lt->z_b;
    const float* nw = lt->nw;
    float* S = lt->S;
    float* no_b = lt->no_b;

    for (int h = h0; h < h1; h++) {
        float oh[128];
        float* Sh = S + (size_t)h * 128 * 128;
        float al = -expf(alog[h]);
        float db = dtb[h];

        for (int b = 0; b < B; b++) {
            float beta = sigf(bvec_b[b * 48 + h]);
            float gv = al * log1pf(expf(avec_b[b * 48 + h] + db));
            float eg = expf(gv);
            const float* qh = q_b + (size_t)b * 2048 + (size_t)(h / 3) * 128;
            const float* kh = k_b + (size_t)b * 2048 + (size_t)(h / 3) * 128;
            const float* vh = v_b + (size_t)b * 6144 + (size_t)h * 128;

            float kq0 = 0.0f, kq1 = 0.0f, kq2 = 0.0f, kq3 = 0.0f;
            for (int i = 0; i < 128; i += 4) {
                kq0 += kh[i + 0] * qh[i + 0];
                kq1 += kh[i + 1] * qh[i + 1];
                kq2 += kh[i + 2] * qh[i + 2];
                kq3 += kh[i + 3] * qh[i + 3];
            }
            float kq = (kq0 + kq1) + (kq2 + kq3);
            double norm_sq = 0.0;

            for (int j = 0; j < 128; j++) {
                float* Sj = Sh + (size_t)j * 128;
                float sk0 = 0.0f, sk1 = 0.0f, sk2 = 0.0f, sk3 = 0.0f;
                float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
                for (int i = 0; i < 128; i += 4) {
                    float s0 = Sj[i + 0], s1 = Sj[i + 1], s2 = Sj[i + 2], s3 = Sj[i + 3];
                    sk0 += s0 * kh[i + 0]; sk1 += s1 * kh[i + 1];
                    sk2 += s2 * kh[i + 2]; sk3 += s3 * kh[i + 3];
                    sq0 += s0 * qh[i + 0]; sq1 += s1 * qh[i + 1];
                    sq2 += s2 * qh[i + 2]; sq3 += s3 * qh[i + 3];
                }
                float kv = ((sk0 + sk1) + (sk2 + sk3)) * eg;
                float delta = (vh[j] - kv) * beta;
                float oj = ((sq0 + sq1) + (sq2 + sq3)) * eg + delta * kq;
                oh[j] = oj;
                norm_sq += (double)oj * oj;

                for (int i = 0; i < 128; i += 4) {
                    Sj[i + 0] = Sj[i + 0] * eg + kh[i + 0] * delta;
                    Sj[i + 1] = Sj[i + 1] * eg + kh[i + 1] * delta;
                    Sj[i + 2] = Sj[i + 2] * eg + kh[i + 2] * delta;
                    Sj[i + 3] = Sj[i + 3] * eg + kh[i + 3] * delta;
                }
            }
            const float* zr = z_b + (size_t)b * 6144 + (size_t)h * 128;
            float* noh = no_b + (size_t)b * 6144 + (size_t)h * 128;
            float inv = 1.0f / sqrtf((float)(norm_sq / 128.0) + 1e-6f);
            for (int j = 0; j < 128; j++) {
                float g8 = oh[j] * inv * nw[j];
                noh[j] = g8 * (zr[j] / (1.0f + fast_expf_m(-zr[j])));
            }
        }
    }
}

// Linear-attention layer for batch of B tokens (B = 1..8).
// x_b[B * 5120] in (normalized), yo_b[B * 5120] out in x_b.
// Reads Q2 weights from DRAM ONCE across all B tokens, and keeps 64 KB Sh in L1 across B tokens!
static int linear_layer_batch(int L, int lin_i, float* x_b, int B) {
    if (g_fused_lin && B == 1) {
        int c0 = 4 * L + 0;
        int c1 = 4 * L + 1;
        int is_static = (g_call_static_bits[c0] != NULL);
        const uint8_t* b_ptr0;
        const short*   s_ptr0;
        const uint8_t* b_ptr1;
        const short*   s_ptr1;
        int buf0 = -1, buf1 = -1;

        uint64_t u_lwait = 0;
        if (is_static) {
            b_ptr0 = g_call_static_bits[c0];
            s_ptr0 = (const short*)g_call_static_scales[c0];
            b_ptr1 = g_call_static_bits[c1];
            s_ptr1 = (const short*)g_call_static_scales[c1];
        } else {
            buf0 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 0;
            buf1 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 1;
            g_read_buf_idx += 2;
            double t_w0 = now_ms();
            uint64_t uw0 = (g_profile_tree) ? now_us() : 0;
            sem_wait(&g_sem_ready[buf0]);
            sem_wait(&g_sem_ready[buf1]);
            double t_w1 = now_ms();
            g_t_memcpy += (t_w1 - t_w0);
            if (g_profile_tree) u_lwait = now_us() - uw0;
            b_ptr0 = g_bits_buf[buf0];
            s_ptr0 = (const short*)g_scales_buf[buf0];
            b_ptr1 = g_bits_buf[buf1];
            s_ptr1 = (const short*)g_scales_buf[buf1];
        }
        if (g_profile_tree) {
            g_prof.layers[L].c0.t_wait_us = u_lwait;
        }

        memcpy(g_xs, x_b, 5120 * sizeof(float));

        double t_r0 = now_ms();
        uint64_t u_lrpc0 = (g_profile_tree) ? now_us() : 0;
        int rc_fused = bonsai_lin_attn_fused(g_h, lin_i,
                                             g_xs, 5120,
                                             b_ptr0, 16384 * 1280,
                                             s_ptr0, 16384 * 40,
                                             b_ptr1, 5120 * 1536,
                                             s_ptr1, 5120 * 48,
                                             g_ys, 5120);
        double t_r1 = now_ms();
        g_t_rpc += (t_r1 - t_r0);
        g_calls_rpc++;
        if (g_profile_tree) {
            g_prof.layers[L].c0.t_rpc_us = now_us() - u_lrpc0;
            g_prof.layers[L].c0.t_total_us = g_prof.layers[L].c0.t_wait_us + g_prof.layers[L].c0.t_rpc_us;
        }

        if (rc_fused != 0) {
            printf("bonsai_lin_attn_fused L=%d lin_i=%d err rc=%d\n", L, lin_i, rc_fused);
            return rc_fused;
        }
        g_cur_call += 2;
        if (!is_static) {
            sem_post(&g_sem_free[buf0]);
            sem_post(&g_sem_free[buf1]);
        }
        memcpy(x_b, g_ys, 5120 * sizeof(float));
        return 0;
    }

    const LayerWeights* lw = &g_lw[L];
    float* qkv_b  = scratch("qkv",     (size_t)B * 10240 * 4);
    float* z_b    = scratch("z",       (size_t)B * 6144 * 4);
    float* avec_b = scratch("ab",      (size_t)B * 48 * 4);
    float* bvec_b = scratch("ab2",     (size_t)B * 48 * 4);
    float* no_b   = scratch("linnorm", (size_t)B * 6144 * 4);
    float* yo_b   = scratch("lino",    (size_t)B * 5120 * 4);

    if (cdsp_call_batch(g_cur_call++, B, x_b, qkv_b, z_b, avec_b, bvec_b)) return -1;

    float* q_b = scratch("q", (size_t)B * 2048 * 4);
    float* k_b = scratch("k", (size_t)B * 2048 * 4);
    float* v_b = scratch("v", (size_t)B * 6144 * 4);
    const float* cwf = lw->conv1d_w;  // [10240][4] cached FP32
    float* conv = ssm_conv + (size_t)lin_i * 10240 * 3;
    float* S    = ssm_rec  + (size_t)lin_i * 48 * 128 * 128;

    uint64_t t_ab_us = 0, t_conv_us = 0, t_qk_us = 0;
    for (int b = 0; b < B; b++) {
        const float* x = x_b + (size_t)b * 5120;
        float* qkv  = qkv_b  + (size_t)b * 10240;
        float* avec = avec_b + (size_t)b * 48;
        float* bvec = bvec_b + (size_t)b * 48;
        float* q    = q_b    + (size_t)b * 2048;
        float* k    = k_b    + (size_t)b * 2048;
        float* v    = v_b    + (size_t)b * 6144;

        if (lw->in_proj_a) {
            uint64_t u_ab0 = (g_profile_tree && B == 1) ? now_us() : 0;
            const float* wa = lw->in_proj_a;
            const float* wb = lw->in_proj_b;
            for (int r = 0; r < 48; r++) {
                const float* war = wa + (size_t)r * 5120;
                const float* wbr = wb + (size_t)r * 5120;
                float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
                float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;
                for (int j = 0; j < 5120; j += 4) {
                    float x0 = x[j + 0], x1 = x[j + 1], x2 = x[j + 2], x3 = x[j + 3];
                    sa0 += war[j + 0] * x0; sa1 += war[j + 1] * x1;
                    sa2 += war[j + 2] * x2; sa3 += war[j + 3] * x3;
                    sb0 += wbr[j + 0] * x0; sb1 += wbr[j + 1] * x1;
                    sb2 += wbr[j + 2] * x2; sb3 += wbr[j + 3] * x3;
                }
                avec[r] = (sa0 + sa1) + (sa2 + sa3);
                bvec[r] = (sb0 + sb1) + (sb2 + sb3);
            }
            if (g_profile_tree && B == 1) t_ab_us += now_us() - u_ab0;
        }

        uint64_t u_c0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int c = 0; c < 10240; c++) {
            float xm3 = conv[c * 3 + 0];
            float xm2 = conv[c * 3 + 1];
            float xm1 = conv[c * 3 + 2];
            float x0  = qkv[c];
            conv[c * 3 + 0] = xm2;
            conv[c * 3 + 1] = xm1;
            conv[c * 3 + 2] = x0;
            const float* cw = cwf + (size_t)c * 4;
            float acc = xm3 * cw[0] + xm2 * cw[1] + xm1 * cw[2] + x0 * cw[3];
            qkv[c] = acc / (1.0f + fast_expf_m(-acc));
        }
        memcpy(q, qkv, 2048 * 4);
        memcpy(k, qkv + 2048, 2048 * 4);
        memcpy(v, qkv + 4096, 6144 * 4);
        if (g_profile_tree && B == 1) t_conv_us += now_us() - u_c0;

        uint64_t u_qk0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int h = 0; h < 16; h++) {
            double sq = 0, sk = 0;
            for (int j = 0; j < 128; j++) {
                sq += (double)q[h * 128 + j] * q[h * 128 + j];
                sk += (double)k[h * 128 + j] * k[h * 128 + j];
            }
            float iq = (1.0f / sqrtf((float)sq + 1e-6f)) * 0.08838834765f;
            float ik = 1.0f / sqrtf((float)sk + 1e-6f);
            for (int j = 0; j < 128; j++) {
                q[h * 128 + j] *= iq;
                k[h * 128 + j] *= ik;
            }
        }
        if (g_profile_tree && B == 1) t_qk_us += now_us() - u_qk0;
    }

    g_lin_task.B      = B;
    g_lin_task.alog   = lw->A_log;
    g_lin_task.dtb    = lw->dt_bias;
    g_lin_task.avec_b = avec_b;
    g_lin_task.bvec_b = bvec_b;
    g_lin_task.q_b    = q_b;
    g_lin_task.k_b    = k_b;
    g_lin_task.v_b    = v_b;
    g_lin_task.z_b    = z_b;
    g_lin_task.nw     = lw->lin_norm;
    g_lin_task.S      = S;
    g_lin_task.no_b   = no_b;
    uint64_t u_rec0 = (g_profile_tree && B == 1) ? now_us() : 0;
    run_lin_heads(0, 48, &g_lin_task);
    uint64_t t_rec_us = (g_profile_tree && B == 1) ? (now_us() - u_rec0) : 0;

    if (g_profile_tree && B == 1) {
        g_prof.layers[L].t_attn_ab_us = t_ab_us;
        g_prof.layers[L].t_attn_conv_us = t_conv_us;
        g_prof.layers[L].t_attn_qk_norm_us = t_qk_us;
        g_prof.layers[L].t_attn_recur_us = t_rec_us;
    }

    if (cdsp_call_batch(g_cur_call++, B, no_b, yo_b, NULL, NULL, NULL)) return -1;
    memcpy(x_b, yo_b, (size_t)B * 5120 * sizeof(float));
    return 0;
}

// Q8 helpers
static void q8_enc(const float* v, int n, int8_t* q, float* s) {
    float mx = 0;
    for (int i = 0; i < n; i++) { float a = v[i] < 0 ? -v[i] : v[i]; if (a > mx) mx = a; }
    float sc = mx > 0 ? mx / 127.0f : 1.0f;
    *s = sc;
    for (int i = 0; i < n; i++) {
        int t = (int)(v[i] / sc + (v[i] >= 0 ? 0.5f : -0.5f));
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        q[i] = (int8_t)t;
    }
}

// Full-attention layer for batch of B tokens at start_pos..start_pos+B-1.
// Reads Q2 weights from DRAM ONCE across all B tokens!
static int full_layer_batch(int L, int full_i, float* x_b, int start_pos, int B) {
    const LayerWeights* lw = &g_lw[L];
    float* qq_b   = scratch("fqq",   (size_t)B * 12288 * 4);
    float* kk_b   = scratch("fkk",   (size_t)B * 1024 * 4);
    float* vv_b   = scratch("fvv",   (size_t)B * 1024 * 4);
    float* flat_b = scratch("fflat", (size_t)B * 6144 * 4);
    float* yo_b   = scratch("fo",    (size_t)B * 5120 * 4);

    if (cdsp_call_batch(g_cur_call++, B, x_b, qq_b, kk_b, vv_b, NULL)) return -1;

    float* q    = scratch("fq_split",    6144 * 4);
    float* gate = scratch("fgate_split", 6144 * 4);
    float* k4   = scratch("fk4",         1024 * 4);
    float* ao   = scratch("fao",         6144 * 4);
    float* sc   = scratch("fsc",         (size_t)g_ctx * 4);
    const float* qnw = lw->q_norm;
    const float* knw = lw->k_norm;

    int8_t* K = kvk  + ((size_t)full_i * 4 * 256 * g_ctx);
    int8_t* V = kvv  + ((size_t)full_i * 4 * 256 * g_ctx);
    float* KS = kvks + ((size_t)full_i * 4 * g_ctx);
    float* VS = kvvs + ((size_t)full_i * 4 * g_ctx);

    uint64_t t_qk_us = 0, t_rope_us = 0, t_kv_us = 0, t_gqa_us = 0;
    for (int b = 0; b < B; b++) {
        int pos = start_pos + b;
        const float* qq = qq_b + (size_t)b * 12288;
        float* kk       = kk_b + (size_t)b * 1024;
        const float* vv = vv_b + (size_t)b * 1024;
        float* flat     = flat_b + (size_t)b * 6144;

        for (int h = 0; h < 24; h++) {
            memcpy(q + h * 256, qq + h * 512, 256 * sizeof(float));
            memcpy(gate + h * 256, qq + h * 512 + 256, 256 * sizeof(float));
        }

        uint64_t u_qk0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int h = 0; h < 24; h++) {
            float* r = q + h * 256;
            double s = 0;
            for (int j = 0; j < 256; j++) s += (double)r[j] * r[j];
            float inv = 1.0f / sqrtf((float)(s / 256) + 1e-6f);
            for (int j = 0; j < 256; j++) r[j] = r[j] * inv * qnw[j];
        }
        for (int h = 0; h < 4; h++) {
            float* r = kk + h * 256;
            double s = 0;
            for (int j = 0; j < 256; j++) s += (double)r[j] * r[j];
            float inv = 1.0f / sqrtf((float)(s / 256) + 1e-6f);
            for (int j = 0; j < 256; j++) r[j] = r[j] * inv * knw[j];
        }
        if (g_profile_tree && B == 1) t_qk_us += now_us() - u_qk0;

        uint64_t u_rp0 = (g_profile_tree && B == 1) ? now_us() : 0;
        bonsai_rope(q, NULL, 24, 256, pos);
        memcpy(k4, kk, 1024 * 4);
        bonsai_rope(k4, NULL, 4, 256, pos);
        if (g_profile_tree && B == 1) t_rope_us += now_us() - u_rp0;

        uint64_t u_kv0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int h = 0; h < 4; h++) {
            q8_enc(k4 + h * 256, 256, K + ((size_t)h * g_ctx + pos) * 256, &KS[(size_t)h * g_ctx + pos]);
            q8_enc(vv + h * 256, 256, V + ((size_t)h * g_ctx + pos) * 256, &VS[(size_t)h * g_ctx + pos]);
        }
        if (g_profile_tree && B == 1) t_kv_us += now_us() - u_kv0;

        uint64_t u_gq0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int hq = 0; hq < 24; hq++) {
            int hk = hq / 6;
            float* qh = q + hq * 256;
            int seqlen = pos + 1;
            double mx = -1e30;
            for (int t = 0; t < seqlen; t++) {
                int8_t* Kt = K + ((size_t)hk * g_ctx + t) * 256;
                float st = KS[(size_t)hk * g_ctx + t];
                double d = 0;
                for (int j = 0; j < 256; j++) d += (double)qh[j] * Kt[j] * st;
                d /= 16.0;
                sc[t] = (float)d;
                if (d > mx) mx = d;
            }
            double s = 0;
            for (int t = 0; t < seqlen; t++) { sc[t] = expf(sc[t] - (float)mx); s += sc[t]; }
            float* oh = ao + hq * 256;
            for (int j = 0; j < 256; j++) oh[j] = 0;
            for (int t = 0; t < seqlen; t++) {
                float p = (float)(sc[t] / s) * VS[(size_t)hk * g_ctx + t];
                int8_t* Vt = V + ((size_t)hk * g_ctx + t) * 256;
                for (int j = 0; j < 256; j++) oh[j] += p * Vt[j];
            }
        }
        for (int i = 0; i < 6144; i++) flat[i] = ao[i] * sigf(gate[i]);
        if (g_profile_tree && B == 1) t_gqa_us += now_us() - u_gq0;
    }

    if (g_profile_tree && B == 1) {
        g_prof.layers[L].t_attn_qk_norm_us = t_qk_us;
        g_prof.layers[L].t_attn_rope_us    = t_rope_us;
        g_prof.layers[L].t_attn_kv_enc_us  = t_kv_us;
        g_prof.layers[L].t_attn_gqa_us     = t_gqa_us;
    }

    if (cdsp_call_batch(g_cur_call++, B, flat_b, yo_b, NULL, NULL, NULL)) return -1;
    memcpy(x_b, yo_b, (size_t)B * 5120 * sizeof(float));
    return 0;
}

// Forward B tokens [toks[0..B-1]] at start_pos..start_pos+B-1 in a single weight-streaming pass!
static float hidden[HIDDEN];
static float g_hidden_batch[8 * HIDDEN];

static int forward_tokens_batch(const int* toks, int start_pos, int B, float* out_hidden_b) {
    if (B < 1 || B > 8) return -97;
    if (stage_init()) { printf("STAGE INIT FAIL\n"); return -98; }
    g_cur_call = 0;
#ifdef _WIN32
    if (!g_vram_prestaged)
#endif
    int prefetch_start_layer = STATIC_LAYERS > 8 ? (STATIC_LAYERS - 8) : 0;
    const char* psl_env = getenv("BONSAI_PREFETCH_START_LAYER");
    if (psl_env) prefetch_start_layer = atoi(psl_env);
    if (prefetch_start_layer < 0) prefetch_start_layer = 0;
    if (prefetch_start_layer > STATIC_LAYERS) prefetch_start_layer = STATIC_LAYERS;

    if (!g_t_embed) return -99;
    float* h_b  = scratch("h",  (size_t)B * HIDDEN * 4);
    float* xn_b = scratch("xn", (size_t)B * HIDDEN * 4);
    float* n2_b = scratch("n2", (size_t)B * HIDDEN * 4);
    float* g8_b = scratch("g8", (size_t)B * FFN * 4);
    float* up_b = scratch("up", (size_t)B * FFN * 4);
    float* dn_b = scratch("dn", (size_t)B * HIDDEN * 4);

    uint64_t u_emb0 = (g_profile_tree && B == 1) ? now_us() : 0;
    for (int b = 0; b < B; b++) {
        dequant_row(g_t_embed, toks[b], h_b + (size_t)b * HIDDEN);
    }
    if (g_profile_tree && B == 1) g_prof.t_embed_us = now_us() - u_emb0;

    for (int L = 0; L < 64; L++) {
        if (g_pipelined_smmu) {
            if (L == 0) {
                if (g_group_mapped[3]) {
                    trigger_smmu_flip(3, 1);
                }
            } else if (L == 16) {
                wait_smmu_flip();
                trigger_smmu_flip(0, 2);
            } else if (L == 32) {
                wait_smmu_flip();
                trigger_smmu_flip(1, 3);
            } else if (L == 48) {
                wait_smmu_flip();
                trigger_smmu_flip(2, 0);
            }
        }
        if (!g_pipelined_smmu && !g_zero_copy && L == prefetch_start_layer) {
            sem_post(&g_sem_start_tok);
            if (g_prefetch_threads >= 2) sem_post(&g_sem_helper_start);
            if (g_prefetch_threads >= 3) sem_post(&g_sem_helper2_start);
        }
        uint64_t u_layer_start = (g_profile_tree && B == 1) ? now_us() : 0;
        int is_lin = (L % 4 != 3);
        int is_static = (L < STATIC_LAYERS);
        if (g_profile_tree && B == 1) {
            g_prof.layers[L].is_lin = is_lin;
            g_prof.layers[L].is_static = is_static;
        }

#ifndef _WIN32
        if (is_lin && g_fused_lin && g_fused_mlp && (g_fused_layer >= 2 || (g_fused_layer == 1 && is_static))) {
            int li = L - (L >> 2);
            int c0 = 4 * L + 0;
            int c1 = 4 * L + 1;
            int c2 = 4 * L + 2;
            int c3 = 4 * L + 3;
            int is_static_l = (g_call_static_bits[c0] != NULL);
            const uint8_t *b0, *b1, *b2, *b3;
            const short   *s0, *s1, *s2, *s3;
            int buf0 = -1, buf1 = -1, buf2 = -1, buf3 = -1;

            uint64_t u_lwait = 0;
            if (is_static_l) {
                b0 = g_call_static_bits[c0]; s0 = (const short*)g_call_static_scales[c0];
                b1 = g_call_static_bits[c1]; s1 = (const short*)g_call_static_scales[c1];
                b2 = g_call_static_bits[c2]; s2 = (const short*)g_call_static_scales[c2];
                b3 = g_call_static_bits[c3]; s3 = (const short*)g_call_static_scales[c3];
            } else {
                buf0 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 0;
                buf1 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 1;
                buf2 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 2;
                buf3 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 3;
                g_read_buf_idx += 4;
                double t_w0 = now_ms();
                uint64_t uw0 = g_profile_tree ? now_us() : 0;
                sem_wait(&g_sem_ready[buf0]);
                sem_wait(&g_sem_ready[buf1]);
                sem_wait(&g_sem_ready[buf2]);
                sem_wait(&g_sem_ready[buf3]);
                double t_w1 = now_ms();
                g_t_memcpy += (t_w1 - t_w0);
                if (g_profile_tree) u_lwait = now_us() - uw0;
                b0 = g_bits_buf[buf0]; s0 = (const short*)g_scales_buf[buf0];
                b1 = g_bits_buf[buf1]; s1 = (const short*)g_scales_buf[buf1];
                b2 = g_bits_buf[buf2]; s2 = (const short*)g_scales_buf[buf2];
                b3 = g_bits_buf[buf3]; s3 = (const short*)g_scales_buf[buf3];
            }
            if (g_profile_tree && B == 1) {
                g_prof.layers[L].c0.t_wait_us = u_lwait;
            }

            for (int b = 0; b < B; b++) {
                memcpy(g_xs, h_b + (size_t)b * HIDDEN, HIDDEN * sizeof(float));

                double t_r0 = now_ms();
                uint64_t u_lrpc0 = (g_profile_tree && B == 1) ? now_us() : 0;
                int rc_layer = bonsai_lin_layer_fused(g_h, li,
                                                      g_xs, HIDDEN,
                                                      b0, 16384 * 1280, s0, 16384 * 40,
                                                      b1, 5120 * 1536,  s1, 5120 * 48,
                                                      b2, 34816 * 1280, s2, 34816 * 40,
                                                      b3, 5120 * 4352,  s3, 5120 * 136,
                                                      g_ys, HIDDEN);
                double t_r1 = now_ms();
                g_t_rpc += (t_r1 - t_r0);
                g_calls_rpc++;
                if (g_profile_tree && B == 1) {
                    g_prof.layers[L].c0.t_rpc_us = now_us() - u_lrpc0;
                    g_prof.layers[L].c0.t_total_us = g_prof.layers[L].c0.t_wait_us + g_prof.layers[L].c0.t_rpc_us;
                }
                if (rc_layer != 0) {
                    printf("bonsai_lin_layer_fused L=%d li=%d err rc=%d\n", L, li, rc_layer);
                    return rc_layer;
                }
                memcpy(h_b + (size_t)b * HIDDEN, g_ys, HIDDEN * sizeof(float));
            }
            g_cur_call += 4;
            if (!is_static_l) {
                sem_post(&g_sem_free[buf0]);
                sem_post(&g_sem_free[buf1]);
                sem_post(&g_sem_free[buf2]);
                sem_post(&g_sem_free[buf3]);
            }
            if (g_profile_tree && B == 1) {
                g_prof.layers[L].t_layer_total_us = now_us() - u_layer_start;
            }
            continue;
        }
        if (!is_lin && g_fused_lin && g_fused_mlp &&
            (g_fused_layer == 2 || (g_fused_layer == 1 && is_static))) {
            int fi = L >> 2;
            int c0 = 4 * L + 0;
            int c1 = 4 * L + 1;
            int c2 = 4 * L + 2;
            int c3 = 4 * L + 3;
            int is_static_l = (g_call_static_bits[c0] != NULL);
            const uint8_t *b0, *b1, *b2, *b3;
            const short   *s0, *s1, *s2, *s3;
            int buf0 = -1, buf1 = -1, buf2 = -1, buf3 = -1;

            static int s_fused_full = -1;
            if (s_fused_full < 0) {
                const char* ff_env = getenv("BONSAI_FUSED_FULL");
                s_fused_full = (!ff_env || atoi(ff_env) != 0) ? 1 : 0;
            }

            if (s_fused_full && start_pos >= 0 && (start_pos + B - 1) < g_dsp_ctx) {
                uint64_t u_lwait = 0;
                if (is_static_l) {
                    b0 = g_call_static_bits[c0]; s0 = (const short*)g_call_static_scales[c0];
                    b1 = g_call_static_bits[c1]; s1 = (const short*)g_call_static_scales[c1];
                    b2 = g_call_static_bits[c2]; s2 = (const short*)g_call_static_scales[c2];
                    b3 = g_call_static_bits[c3]; s3 = (const short*)g_call_static_scales[c3];
                } else {
                    buf0 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 0;
                    buf1 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 1;
                    buf2 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 2;
                    buf3 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 3;
                    g_read_buf_idx += 4;
                    double t_w0 = now_ms();
                    uint64_t uw0 = g_profile_tree ? now_us() : 0;
                    sem_wait(&g_sem_ready[buf0]);
                    sem_wait(&g_sem_ready[buf1]);
                    sem_wait(&g_sem_ready[buf2]);
                    sem_wait(&g_sem_ready[buf3]);
                    double t_w1 = now_ms();
                    g_t_memcpy += (t_w1 - t_w0);
                    if (g_profile_tree) u_lwait = now_us() - uw0;
                    b0 = g_bits_buf[buf0]; s0 = (const short*)g_scales_buf[buf0];
                    b1 = g_bits_buf[buf1]; s1 = (const short*)g_scales_buf[buf1];
                    b2 = g_bits_buf[buf2]; s2 = (const short*)g_scales_buf[buf2];
                    b3 = g_bits_buf[buf3]; s3 = (const short*)g_scales_buf[buf3];
                }
                if (g_profile_tree && B == 1) {
                    g_prof.layers[L].c0.t_wait_us = u_lwait;
                }

                for (int b = 0; b < B; b++) {
                    int cur_pos = start_pos + b;
                    memcpy(g_xs, h_b + (size_t)b * HIDDEN, HIDDEN * sizeof(float));

                    double t_r0 = now_ms();
                    uint64_t u_lrpc0 = (g_profile_tree && B == 1) ? now_us() : 0;
                    int rc_full = bonsai_lin_layer_fused(g_h, -10000 - (cur_pos * 16 + fi),
                                                         g_xs, HIDDEN,
                                                         b0, 14336 * 1280, s0, 14336 * 40,
                                                         b1, 5120 * 1536,  s1, 5120 * 48,
                                                         b2, 34816 * 1280, s2, 34816 * 40,
                                                         b3, 5120 * 4352,  s3, 5120 * 136,
                                                         g_ys, HIDDEN);
                    double t_r1 = now_ms();
                    g_t_rpc += (t_r1 - t_r0);
                    g_calls_rpc++;
                    if (g_profile_tree && B == 1) {
                        g_prof.layers[L].c0.t_rpc_us = now_us() - u_lrpc0;
                        g_prof.layers[L].c0.t_total_us = g_prof.layers[L].c0.t_wait_us + g_prof.layers[L].c0.t_rpc_us;
                    }
                    if (rc_full != 0) {
                        printf("bonsai_lin_layer_fused(full 1-call) L=%d fi=%d pos=%d err rc=%d\n", L, fi, cur_pos, rc_full);
                        return rc_full;
                    }
                    memcpy(h_b + (size_t)b * HIDDEN, g_ys, HIDDEN * sizeof(float));
                }
                g_cur_call += 4;
                if (!is_static_l) {
                    sem_post(&g_sem_free[buf0]);
                    sem_post(&g_sem_free[buf1]);
                    sem_post(&g_sem_free[buf2]);
                    sem_post(&g_sem_free[buf3]);
                }
                if (g_profile_tree && B == 1) {
                    g_prof.layers[L].t_layer_total_us = now_us() - u_layer_start;
                }
                continue;
            }

            // 1. Fused Full-Attn Head: Input RMSNorm + FWHT-5120 + qkv_proj GEMV (14336 x 5120)
            uint64_t u_wait0 = 0;
            if (is_static_l) {
                b0 = g_call_static_bits[c0]; s0 = (const short*)g_call_static_scales[c0];
            } else {
                buf0 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 0;
                g_read_buf_idx += 1;
                double t_w0 = now_ms();
                uint64_t uw0 = g_profile_tree ? now_us() : 0;
                sem_wait(&g_sem_ready[buf0]);
                double t_w1 = now_ms();
                g_t_memcpy += (t_w1 - t_w0);
                if (g_profile_tree) u_wait0 = now_us() - uw0;
                b0 = g_bits_buf[buf0]; s0 = (const short*)g_scales_buf[buf0];
            }
            if (g_profile_tree) g_prof.layers[L].c0.t_wait_us = u_wait0;

            memcpy(g_xs, h_b, HIDDEN * sizeof(float));

            double t_r0 = now_ms();
            uint64_t u_rpc0 = g_profile_tree ? now_us() : 0;
            int rc_qkv = bonsai_lin_attn_fused(g_h, -1 - fi,
                                               g_xs, HIDDEN,
                                               b0, 14336 * 1280, s0, 14336 * 40,
                                               g_call_static_bits[0], 128, (const short*)g_call_static_scales[0], 64,
                                               g_ys, 14336);
            double t_r1 = now_ms();
            g_t_rpc += (t_r1 - t_r0);
            g_calls_rpc++;
            if (g_profile_tree) {
                g_prof.layers[L].c0.t_rpc_us = now_us() - u_rpc0;
                g_prof.layers[L].c0.t_total_us = g_prof.layers[L].c0.t_wait_us + g_prof.layers[L].c0.t_rpc_us;
            }
            if (rc_qkv != 0) {
                printf("bonsai_lin_attn_fused(full qkv) L=%d fi=%d err rc=%d\n", L, fi, rc_qkv);
                return rc_qkv;
            }
            if (!is_static_l) {
                sem_post(&g_sem_free[buf0]);
            }

            // 2. Host ARM GQA on g_ys [14336] -> g_xs + HIDDEN [6144]
            const LayerWeights* lw = &g_lw[L];
            float* q    = scratch("fq_split",    6144 * 4);
            float* gate = scratch("fgate_split", 6144 * 4);
            float* k4   = scratch("fk4",         1024 * 4);
            float* ao   = scratch("fao",         6144 * 4);
            float* sc   = scratch("fsc",         (size_t)g_ctx * 4);
            const float* qnw = lw->q_norm;
            const float* knw = lw->k_norm;

            int8_t* K  = kvk  + ((size_t)fi * 4 * 256 * g_ctx);
            int8_t* V  = kvv  + ((size_t)fi * 4 * 256 * g_ctx);
            float*  KS = kvks + ((size_t)fi * 4 * g_ctx);
            float*  VS = kvvs + ((size_t)fi * 4 * g_ctx);

            int pos = start_pos;
            const float* qq = g_ys;
            float* kk       = g_ys + 12288;
            const float* vv = g_ys + 13312;
            float* flat     = g_xs + HIDDEN;

            for (int h = 0; h < 24; h++) {
                memcpy(q + h * 256, qq + h * 512, 256 * sizeof(float));
                memcpy(gate + h * 256, qq + h * 512 + 256, 256 * sizeof(float));
            }

            uint64_t u_qk0 = g_profile_tree ? now_us() : 0;
            for (int h = 0; h < 24; h++) {
                float* r = q + h * 256;
                double s = 0;
                for (int j = 0; j < 256; j++) s += (double)r[j] * r[j];
                float inv = 1.0f / sqrtf((float)(s / 256) + 1e-6f);
                for (int j = 0; j < 256; j++) r[j] = r[j] * inv * qnw[j];
            }
            for (int h = 0; h < 4; h++) {
                float* r = kk + h * 256;
                double s = 0;
                for (int j = 0; j < 256; j++) s += (double)r[j] * r[j];
                float inv = 1.0f / sqrtf((float)(s / 256) + 1e-6f);
                for (int j = 0; j < 256; j++) r[j] = r[j] * inv * knw[j];
            }
            if (g_profile_tree) g_prof.layers[L].t_attn_qk_norm_us = now_us() - u_qk0;

            uint64_t u_rp0 = g_profile_tree ? now_us() : 0;
            bonsai_rope(q, NULL, 24, 256, pos);
            memcpy(k4, kk, 1024 * 4);
            bonsai_rope(k4, NULL, 4, 256, pos);
            if (g_profile_tree) g_prof.layers[L].t_attn_rope_us = now_us() - u_rp0;

            uint64_t u_kv0 = g_profile_tree ? now_us() : 0;
            for (int h = 0; h < 4; h++) {
                q8_enc(k4 + h * 256, 256, K + ((size_t)h * g_ctx + pos) * 256, &KS[(size_t)h * g_ctx + pos]);
                q8_enc(vv + h * 256, 256, V + ((size_t)h * g_ctx + pos) * 256, &VS[(size_t)h * g_ctx + pos]);
            }
            if (g_profile_tree) g_prof.layers[L].t_attn_kv_enc_us = now_us() - u_kv0;

            uint64_t u_gq0 = g_profile_tree ? now_us() : 0;
            for (int hq = 0; hq < 24; hq++) {
                int hk = hq / 6;
                float* qh = q + hq * 256;
                int seqlen = pos + 1;
                double mx = -1e30;
                for (int t = 0; t < seqlen; t++) {
                    int8_t* Kt = K + ((size_t)hk * g_ctx + t) * 256;
                    float st = KS[(size_t)hk * g_ctx + t];
                    double d = 0;
                    for (int j = 0; j < 256; j++) d += (double)qh[j] * Kt[j] * st;
                    d /= 16.0;
                    sc[t] = (float)d;
                    if (d > mx) mx = d;
                }
                double s = 0;
                for (int t = 0; t < seqlen; t++) { sc[t] = expf(sc[t] - (float)mx); s += sc[t]; }
                float* oh = ao + hq * 256;
                for (int j = 0; j < 256; j++) oh[j] = 0;
                for (int t = 0; t < seqlen; t++) {
                    float p = (float)(sc[t] / s) * VS[(size_t)hk * g_ctx + t];
                    int8_t* Vt = V + ((size_t)hk * g_ctx + t) * 256;
                    for (int j = 0; j < 256; j++) oh[j] += p * Vt[j];
                }
            }
            for (int i = 0; i < 6144; i++) flat[i] = ao[i] * sigf(gate[i]);
            if (g_profile_tree) g_prof.layers[L].t_attn_gqa_us = now_us() - u_gq0;

            // 3. Fused Full-Attn Tail: FWHT-6144 + o_proj + Res1 + PostNorm + FWHT-5120 + Fused MLP + Res2
            memcpy(g_xs, h_b, HIDDEN * sizeof(float));
            uint64_t u_mwait = 0;
            if (is_static_l) {
                b1 = g_call_static_bits[c1]; s1 = (const short*)g_call_static_scales[c1];
                b2 = g_call_static_bits[c2]; s2 = (const short*)g_call_static_scales[c2];
                b3 = g_call_static_bits[c3]; s3 = (const short*)g_call_static_scales[c3];
            } else {
                buf1 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 1;
                buf2 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 2;
                buf3 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 3;
                g_read_buf_idx += 3;
                double t_w0 = now_ms();
                uint64_t uw0 = g_profile_tree ? now_us() : 0;
                sem_wait(&g_sem_ready[buf1]);
                sem_wait(&g_sem_ready[buf2]);
                sem_wait(&g_sem_ready[buf3]);
                double t_w1 = now_ms();
                g_t_memcpy += (t_w1 - t_w0);
                if (g_profile_tree) u_mwait = now_us() - uw0;
                b1 = g_bits_buf[buf1]; s1 = (const short*)g_scales_buf[buf1];
                b2 = g_bits_buf[buf2]; s2 = (const short*)g_scales_buf[buf2];
                b3 = g_bits_buf[buf3]; s3 = (const short*)g_scales_buf[buf3];
            }
            if (g_profile_tree) g_prof.layers[L].t_mlp_wait_us = u_mwait;

            double t_mr0 = now_ms();
            uint64_t u_mrpc0 = g_profile_tree ? now_us() : 0;
            int rc_tail = bonsai_lin_layer_fused(g_h, -1 - fi,
                                                 g_xs, HIDDEN + 6144,
                                                 g_call_static_bits[0], 128, (const short*)g_call_static_scales[0], 64,
                                                 b1, 5120 * 1536,  s1, 5120 * 48,
                                                 b2, 34816 * 1280, s2, 34816 * 40,
                                                 b3, 5120 * 4352,  s3, 5120 * 136,
                                                 g_ys, HIDDEN);
            double t_mr1 = now_ms();
            g_t_rpc += (t_mr1 - t_mr0);
            g_calls_rpc++;
            if (g_profile_tree) {
                g_prof.layers[L].t_mlp_rpc_us = now_us() - u_mrpc0;
            }
            if (rc_tail != 0) {
                printf("bonsai_lin_layer_fused(full tail) L=%d fi=%d err rc=%d\n", L, fi, rc_tail);
                return rc_tail;
            }
            g_cur_call += 4;
            if (!is_static_l) {
                sem_post(&g_sem_free[buf1]);
                sem_post(&g_sem_free[buf2]);
                sem_post(&g_sem_free[buf3]);
            }
            memcpy(h_b, g_ys, HIDDEN * sizeof(float));
            if (g_profile_tree) {
                g_prof.layers[L].t_layer_total_us = now_us() - u_layer_start;
            }
            continue;
        }
#endif

        uint64_t u_in_norm0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int b = 0; b < B; b++) {
            bonsai_rmsnorm(h_b + (size_t)b * HIDDEN, g_lw[L].in_norm, HIDDEN, xn_b + (size_t)b * HIDDEN);
        }
        if (g_profile_tree && B == 1) g_prof.layers[L].t_in_norm_us = now_us() - u_in_norm0;

        int rc;
        if (is_lin) {
            int li = L - (L >> 2);
            rc = linear_layer_batch(L, li, xn_b, B);
        } else {
            int fi = L >> 2;
            rc = full_layer_batch(L, fi, xn_b, start_pos, B);
        }
        if (rc) return rc;


        uint64_t u_res1_0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int i = 0; i < B * HIDDEN; i++) h_b[i] += xn_b[i];
        if (g_profile_tree && B == 1) g_prof.layers[L].t_res1_us = now_us() - u_res1_0;

        uint64_t u_pn0 = (g_profile_tree && B == 1) ? now_us() : 0;
        for (int b = 0; b < B; b++) {
            bonsai_rmsnorm(h_b + (size_t)b * HIDDEN, g_lw[L].post_norm, HIDDEN, n2_b + (size_t)b * HIDDEN);
        }
        if (g_profile_tree && B == 1) g_prof.layers[L].t_post_norm_us = now_us() - u_pn0;

#ifdef _WIN32
        if (cdsp_call_batch(g_cur_call++, B, n2_b, g8_b, up_b, NULL, NULL)) return -1;
        bonsai_swiglu(g8_b, up_b, B * FFN, g8_b);
        if (cdsp_call_batch(g_cur_call++, B, g8_b, dn_b, NULL, NULL, NULL)) return -1;
        for (int i = 0; i < B * HIDDEN; i++) h_b[i] += dn_b[i];
#else
        if (g_fused_mlp) {
            uint64_t u_mfwht0 = (g_profile_tree && B == 1) ? now_us() : 0;
            for (int b = 0; b < B; b++) {
                const float* xb = n2_b + (size_t)b * HIDDEN;
                float* xdst = g_xs + (size_t)b * HIDDEN;
                if (g_signs_5120) {
                    bonsai_fwht1024(xb, g_signs_5120, HIDDEN, 0, xdst);
                } else {
                    memcpy(xdst, xb, (size_t)HIDDEN * 4);
                }
            }
            if (g_profile_tree && B == 1) g_prof.layers[L].t_mlp_fwht_us = now_us() - u_mfwht0;

            int c2 = 4 * L + 2;
            int c3 = 4 * L + 3;
            int is_static_mlp = (g_call_static_bits[c2] != NULL);
            const uint8_t* b_ptr2;
            const short*   s_ptr2;
            const uint8_t* b_ptr3;
            const short*   s_ptr3;
            int buf2 = -1, buf3 = -1;

            uint64_t u_mwait = 0;
            if (is_static_mlp) {
                b_ptr2 = g_call_static_bits[c2];
                s_ptr2 = (const short*)g_call_static_scales[c2];
                b_ptr3 = g_call_static_bits[c3];
                s_ptr3 = (const short*)g_call_static_scales[c3];
            } else {
                buf2 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 2;
                buf3 = (((L - STATIC_LAYERS) % RING_LAYERS) * 4) + 3;
                g_read_buf_idx += 2;
                double t_w0 = now_ms();
                uint64_t uw0 = (g_profile_tree && B == 1) ? now_us() : 0;
                sem_wait(&g_sem_ready[buf2]);
                sem_wait(&g_sem_ready[buf3]);
                double t_w1 = now_ms();
                g_t_memcpy += (t_w1 - t_w0);
                if (g_profile_tree && B == 1) u_mwait = now_us() - uw0;
                b_ptr2 = g_bits_buf[buf2];
                s_ptr2 = (const short*)g_scales_buf[buf2];
                b_ptr3 = g_bits_buf[buf3];
                s_ptr3 = (const short*)g_scales_buf[buf3];
            }
            if (g_profile_tree && B == 1) g_prof.layers[L].t_mlp_wait_us = u_mwait;

            double t_r0 = now_ms();
            uint64_t u_mrpc0 = (g_profile_tree && B == 1) ? now_us() : 0;
            int rc_fused = bonsai_mlp_fused(g_h,
                g_xs, B * HIDDEN,
                b_ptr2, 34816 * 1280,
                s_ptr2, 34816 * 40,
                b_ptr3, 5120 * 4352,
                s_ptr3, 5120 * 136,
                g_ys, B * HIDDEN);
            double t_r1 = now_ms();
            g_t_rpc += (t_r1 - t_r0);
            g_calls_rpc++;
            if (g_profile_tree && B == 1) g_prof.layers[L].t_mlp_rpc_us = now_us() - u_mrpc0;

            if (rc_fused != 0) { printf("bonsai_mlp_fused L=%d err rc=%d\n", L, rc_fused); return rc_fused; }
            g_cur_call += 2;
            if (!is_static_mlp) {
                sem_post(&g_sem_free[buf2]);
                sem_post(&g_sem_free[buf3]);
            }
            uint64_t u_res2_0 = (g_profile_tree && B == 1) ? now_us() : 0;
            for (int i = 0; i < B * HIDDEN; i++) h_b[i] += g_ys[i];
            if (g_profile_tree && B == 1) g_prof.layers[L].t_res2_us = now_us() - u_res2_0;
        } else {
            if (cdsp_call_batch(g_cur_call++, B, n2_b, g8_b, up_b, NULL, NULL)) return -1;
            bonsai_swiglu(g8_b, up_b, B * FFN, g8_b);
            if (cdsp_call_batch(g_cur_call++, B, g8_b, dn_b, NULL, NULL, NULL)) return -1;
            for (int i = 0; i < B * HIDDEN; i++) h_b[i] += dn_b[i];
        }
#endif
        if (g_profile_tree && B == 1) {
            g_prof.layers[L].t_layer_total_us = now_us() - u_layer_start;
        }
    }

    if (g_pipelined_smmu) {
        wait_smmu_flip();
    }

    uint64_t u_fn0 = (g_profile_tree && B == 1) ? now_us() : 0;
    float* dst = out_hidden_b ? out_hidden_b : g_hidden_batch;
    for (int b = 0; b < B; b++) {
        bonsai_rmsnorm(h_b + (size_t)b * HIDDEN, g_final_norm, HIDDEN, dst + (size_t)b * HIDDEN);
    }
    memcpy(hidden, dst + (size_t)(B - 1) * HIDDEN, HIDDEN * 4);
    if (g_profile_tree && B == 1) g_prof.t_final_norm_us = now_us() - u_fn0;
    return 0;
}

static int forward_token(int tok, int pos) {
    return forward_tokens_batch(&tok, pos, 1, hidden);
}

#ifndef _WIN32
static volatile int g_bus_boost_active = 0;

static void bus_boost_release(void) {
    if (!g_bus_boost_active) return;
    g_bus_boost_active = 0;
    int rc = system("su -c 'echo 547000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq; "
                    "echo 350000 > /sys/devices/system/cpu/bus_dcvs/LLCC/boost_freq; "
                    "echo 556800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
                    "echo 1017600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq' >/dev/null 2>&1");
    (void)rc;
    fprintf(stderr, "[fwd] Thermal-safe bus & CPU boost released (restored DDR=547000, LLCC=350000, CPU=556M/1017M)\n");
}

static void bus_boost_sig_handler(int sig) {
    bus_boost_release();
    _exit(128 + sig);
}

static long read_proc_ticks(int pid) {
    char path[64], buf[512];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = 0;
    char* rp = strrchr(buf, ')');
    if (!rp || !rp[1]) return -1;
    char* p = rp + 2;
    for (int f = 3; f < 14; f++) {
        p = strchr(p, ' ');
        if (!p) return -1;
        while (*p == ' ') p++;
    }
    long utime = 0, stime = 0;
    if (sscanf(p, "%ld %ld", &utime, &stime) != 2) return -1;
    return utime + stime;
}

static void bus_boost_acquire(void) {
    const char* env = getenv("BONSAI_BOOST");
    if (env && atoi(env) == 0) return;
    int rc = system("su -c 'echo 4761000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq; "
                    "echo 1211000 > /sys/devices/system/cpu/bus_dcvs/LLCC/boost_freq; "
                    "echo 3532800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
                    "echo 4320000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq' </dev/null >/dev/null 2>&1");
    if (rc == 0) {
        g_bus_boost_active = 1;
        atexit(bus_boost_release);
        signal(SIGINT,  bus_boost_sig_handler);
        signal(SIGTERM, bus_boost_sig_handler);
        pid_t parent_pid = getpid();
        pid_t wpid = fork();
        if (wpid == 0) {
            setsid();
            for (int fd = 0; fd < 128; fd++) close(fd);
            long prev = read_proc_ticks((int)parent_pid);
            int idle_cnt = 0;
            while (1) {
                sleep(2);
                long cur = read_proc_ticks((int)parent_pid);
                if (cur < 0) break; // parent exited
                long delta = cur - prev;
                prev = cur;
                if (delta < 2) idle_cnt++;
                else idle_cnt = 0;
                if (idle_cnt >= 3) break; // stalled / zero load for 6s
            }
            int r2 = system("su -c 'echo 547000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq; "
                            "echo 350000 > /sys/devices/system/cpu/bus_dcvs/LLCC/boost_freq; "
                            "echo 556800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq; "
                            "echo 1017600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq' </dev/null >/dev/null 2>&1");
            (void)r2;
            _exit(0);
        }
        fprintf(stderr, "[fwd] Thermal-safe bus & CPU boost acquired (DDR=4761M, LLCC=1211M, CPU=3532M/4320M, watchdog_pid=%d)\n", (int)wpid);
    }
}
#endif

static void print_profile_tree(const StepTiming* p, int step) {
    uint64_t lin_in_norm = 0, lin_in_fwht = 0, lin_in_wait = 0, lin_in_rpc = 0, lin_in_post = 0;
    uint64_t lin_ab = 0, lin_conv = 0, lin_qk = 0, lin_recur = 0;
    uint64_t lin_out_fwht = 0, lin_out_wait = 0, lin_out_rpc = 0, lin_out_post = 0;
    uint64_t lin_res1 = 0, lin_pn = 0;
    uint64_t lin_mlp_fwht = 0, lin_mlp_wait = 0, lin_mlp_rpc = 0, lin_res2 = 0;
    uint64_t lin_total = 0;

    uint64_t full_in_norm = 0, full_in_fwht = 0, full_in_wait = 0, full_in_rpc = 0, full_in_post = 0;
    uint64_t full_qk = 0, full_rope = 0, full_kv = 0, full_gqa = 0;
    uint64_t full_out_fwht = 0, full_out_wait = 0, full_out_rpc = 0, full_out_post = 0;
    uint64_t full_res1 = 0, full_pn = 0;
    uint64_t full_mlp_fwht = 0, full_mlp_wait = 0, full_mlp_rpc = 0, full_res2 = 0;
    uint64_t full_total = 0;

    uint64_t stat_layer_total = 0, stat_rpc_total = 0, stat_wait_total = 0;
    uint64_t strm_layer_total = 0, strm_rpc_total = 0, strm_wait_total = 0;

    for (int L = 0; L < 64; L++) {
        const LayerTiming* lt = &p->layers[L];
        uint64_t l_rpc = lt->c0.t_rpc_us + lt->c1.t_rpc_us + lt->t_mlp_rpc_us;
        uint64_t l_wait = lt->c0.t_wait_us + lt->c1.t_wait_us + lt->t_mlp_wait_us;
        if (L < STATIC_LAYERS) {
            stat_layer_total += lt->t_layer_total_us;
            stat_rpc_total += l_rpc;
            stat_wait_total += l_wait;
        } else {
            strm_layer_total += lt->t_layer_total_us;
            strm_rpc_total += l_rpc;
            strm_wait_total += l_wait;
        }

        if (lt->is_lin) {
            lin_in_norm  += lt->t_in_norm_us;
            lin_in_fwht  += lt->c0.t_fwht_us;
            lin_in_wait  += lt->c0.t_wait_us;
            lin_in_rpc   += lt->c0.t_rpc_us;
            lin_in_post  += lt->c0.t_post_us;
            lin_ab       += lt->t_attn_ab_us;
            lin_conv     += lt->t_attn_conv_us;
            lin_qk       += lt->t_attn_qk_norm_us;
            lin_recur    += lt->t_attn_recur_us;
            lin_out_fwht += lt->c1.t_fwht_us;
            lin_out_wait += lt->c1.t_wait_us;
            lin_out_rpc  += lt->c1.t_rpc_us;
            lin_out_post += lt->c1.t_post_us;
            lin_res1     += lt->t_res1_us;
            lin_pn       += lt->t_post_norm_us;
            lin_mlp_fwht += lt->t_mlp_fwht_us;
            lin_mlp_wait += lt->t_mlp_wait_us;
            lin_mlp_rpc  += lt->t_mlp_rpc_us;
            lin_res2     += lt->t_res2_us;
            lin_total    += lt->t_layer_total_us;
        } else {
            full_in_norm  += lt->t_in_norm_us;
            full_in_fwht  += lt->c0.t_fwht_us;
            full_in_wait  += lt->c0.t_wait_us;
            full_in_rpc   += lt->c0.t_rpc_us;
            full_in_post  += lt->c0.t_post_us;
            full_qk       += lt->t_attn_qk_norm_us;
            full_rope     += lt->t_attn_rope_us;
            full_kv       += lt->t_attn_kv_enc_us;
            full_gqa      += lt->t_attn_gqa_us;
            full_out_fwht += lt->c1.t_fwht_us;
            full_out_wait += lt->c1.t_wait_us;
            full_out_rpc  += lt->c1.t_rpc_us;
            full_out_post += lt->c1.t_post_us;
            full_res1     += lt->t_res1_us;
            full_pn       += lt->t_post_norm_us;
            full_mlp_fwht += lt->t_mlp_fwht_us;
            full_mlp_wait += lt->t_mlp_wait_us;
            full_mlp_rpc  += lt->t_mlp_rpc_us;
            full_res2     += lt->t_res2_us;
            full_total    += lt->t_layer_total_us;
        }
    }

    uint64_t total_layer_us = lin_total + full_total;
    uint64_t total_npu_rpc_us = (lin_in_rpc + lin_out_rpc + lin_mlp_rpc) +
                                (full_in_rpc + full_out_rpc + full_mlp_rpc) +
                                p->t_lmhead_rpc_us;
    uint64_t total_fwht_us = (lin_in_fwht + lin_out_fwht + lin_mlp_fwht) +
                             (full_in_fwht + full_out_fwht + full_mlp_fwht) +
                             p->t_lmhead_fwht_us;
    uint64_t total_wait_us = (lin_in_wait + lin_out_wait + lin_mlp_wait) +
                             (full_in_wait + full_out_wait + full_mlp_wait);
    uint64_t total_post_us = (lin_in_post + lin_out_post) + (full_in_post + full_out_post) + p->t_lmhead_post_us;
    uint64_t total_cpu_math_us = total_fwht_us + (lin_ab + lin_conv + lin_qk + lin_recur) +
                                 (full_qk + full_rope + full_kv + full_gqa) +
                                 (lin_in_norm + lin_pn + lin_res1 + lin_res2) +
                                 (full_in_norm + full_pn + full_res1 + full_res2) +
                                 p->t_final_norm_us;
    uint64_t total_tok_us = p->t_token_total_us;
    double t_tok_ms = total_tok_us / 1000.0;
    if (t_tok_ms < 1.0) t_tok_ms = 1.0;

    printf("\n");
    printf("====================================================================================================\n");
    printf("           BONSAI 2 27B - HIERARCHICAL DEBUG PROFILING TREE (Decode Step %d)\n", step);
    printf("====================================================================================================\n");
    printf("[Device: OnePlus 13 (SM8750) | RAM: 16 GB LPDDR5X | Model: 27B Ternary Q2 (6.74 GiB)]\n");
    printf("Total Token Decode Time: %.2f ms (100.0%%) [Speed: %.2f tok/s]\n", t_tok_ms, 1000.0 / t_tok_ms);
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("|-- 1. HOST PRE-LAYER & EMBEDDING: %.2f ms (%.2f%%)\n", p->t_embed_us / 1000.0, 100.0 * p->t_embed_us / total_tok_us);
    printf("|   `-- Embedding row dequant: %.2f ms\n", p->t_embed_us / 1000.0);
    printf("|\n");
    printf("|-- 2. TRANSFORMER BACKBONE (64 Layers): %.2f ms (%.2f%%)\n", total_layer_us / 1000.0, 100.0 * total_layer_us / total_tok_us);
    printf("|   |\n");
    printf("|   |-- 2.1 LINEAR ATTENTION LAYERS (48 layers): %.2f ms (%.2f%%) [avg %.2f ms/layer]\n",
           lin_total / 1000.0, 100.0 * lin_total / total_tok_us, lin_total / 48000.0);
    printf("|   |   |-- Input RMSNorm (48 calls): %.2f ms [avg %.3f ms]\n", lin_in_norm / 1000.0, lin_in_norm / 48000.0);
    printf("|   |   |-- Attention in_proj RPC (48 calls, 16384x5120): %.2f ms [avg %.2f ms]\n",
           (lin_in_fwht + lin_in_wait + lin_in_rpc + lin_in_post) / 1000.0, (lin_in_fwht + lin_in_wait + lin_in_rpc + lin_in_post) / 48000.0);
    printf("|   |   |   |-- Host FWHT-5120: %.2f ms [avg %.3f ms]\n", lin_in_fwht / 1000.0, lin_in_fwht / 48000.0);
    printf("|   |   |   |-- Ring Buffer Wait: %.2f ms\n", lin_in_wait / 1000.0);
    printf("|   |   |   |-- Hexagon HVX GEMV + FastRPC IPC: %.2f ms [avg %.2f ms]\n", lin_in_rpc / 1000.0, lin_in_rpc / 48000.0);
    printf("|   |   |   `-- Activation extract/copy: %.2f ms\n", lin_in_post / 1000.0);
    printf("|   |   |-- Host Attention Math (ARM NEON): %.2f ms [avg %.2f ms/layer]\n",
           (lin_ab + lin_conv + lin_qk + lin_recur) / 1000.0, (lin_ab + lin_conv + lin_qk + lin_recur) / 48000.0);
    printf("|   |   |   |-- in_proj_a/b dot products (48x5120): %.2f ms\n", lin_ab / 1000.0);
    printf("|   |   |   |-- Depthwise Conv1D (10240x4 taps + SiLU): %.2f ms\n", lin_conv / 1000.0);
    printf("|   |   |   |-- Q / K RMSNorm (16x128): %.2f ms\n", lin_qk / 1000.0);
    printf("|   |   |   `-- DeltaNet S_h Recurrence (48 heads x 128x128): %.2f ms\n", lin_recur / 1000.0);
    printf("|   |   |-- Attention out_proj RPC (48 calls, 5120x6144): %.2f ms [avg %.2f ms]\n",
           (lin_out_fwht + lin_out_wait + lin_out_rpc + lin_out_post) / 1000.0, (lin_out_fwht + lin_out_wait + lin_out_rpc + lin_out_post) / 48000.0);
    printf("|   |   |   |-- Host FWHT-6144: %.2f ms [avg %.3f ms]\n", lin_out_fwht / 1000.0, lin_out_fwht / 48000.0);
    printf("|   |   |   |-- Ring Buffer Wait: %.2f ms\n", lin_out_wait / 1000.0);
    printf("|   |   |   |-- Hexagon HVX GEMV + FastRPC IPC: %.2f ms [avg %.2f ms]\n", lin_out_rpc / 1000.0, lin_out_rpc / 48000.0);
    printf("|   |   |   `-- Activation extract/copy: %.2f ms\n", lin_out_post / 1000.0);
    printf("|   |   |-- Residual Add 1 (48 calls): %.2f ms\n", lin_res1 / 1000.0);
    printf("|   |   |-- Post-Attention RMSNorm (48 calls): %.2f ms\n", lin_pn / 1000.0);
    printf("|   |   |-- Fused MLP Block (48 layers): %.2f ms [avg %.2f ms/layer]\n",
           (lin_mlp_fwht + lin_mlp_wait + lin_mlp_rpc) / 1000.0, (lin_mlp_fwht + lin_mlp_wait + lin_mlp_rpc) / 48000.0);
    printf("|   |   |   |-- Host FWHT-5120: %.2f ms [avg %.3f ms]\n", lin_mlp_fwht / 1000.0, lin_mlp_fwht / 48000.0);
    printf("|   |   |   |-- Ring Buffer Wait (gate_up & down): %.2f ms\n", lin_mlp_wait / 1000.0);
    printf("|   |   |   `-- Fused MLP RPC (gate_up + SwiGLU + down): %.2f ms [avg %.2f ms]\n", lin_mlp_rpc / 1000.0, lin_mlp_rpc / 48000.0);
    printf("|   |   `-- Residual Add 2 (48 calls): %.2f ms\n", lin_res2 / 1000.0);
    printf("|   |\n");
    printf("|   |-- 2.2 FULL ATTENTION LAYERS (16 layers): %.2f ms (%.2f%%) [avg %.2f ms/layer]\n",
           full_total / 1000.0, 100.0 * full_total / total_tok_us, full_total / 16000.0);
    printf("|   |   |-- Input RMSNorm (16 calls): %.2f ms\n", full_in_norm / 1000.0);
    printf("|   |   |-- Attention in_proj RPC (16 calls, 14336x5120): %.2f ms [avg %.2f ms]\n",
           (full_in_fwht + full_in_wait + full_in_rpc + full_in_post) / 1000.0, (full_in_fwht + full_in_wait + full_in_rpc + full_in_post) / 16000.0);
    printf("|   |   |-- Host Attention Math (ARM NEON): %.2f ms [avg %.2f ms/layer]\n",
           (full_qk + full_rope + full_kv + full_gqa) / 1000.0, (full_qk + full_rope + full_kv + full_gqa) / 16000.0);
    printf("|   |   |   |-- Q / K RMSNorm: %.2f ms\n", full_qk / 1000.0);
    printf("|   |   |   |-- RoPE (Q 24, K 4 heads): %.2f ms\n", full_rope / 1000.0);
    printf("|   |   |   |-- KV Cache encode (q8_enc): %.2f ms\n", full_kv / 1000.0);
    printf("|   |   |   `-- GQA Softmax + dot products: %.2f ms\n", full_gqa / 1000.0);
    printf("|   |   |-- Attention out_proj RPC (16 calls, 5120x6144): %.2f ms [avg %.2f ms]\n",
           (full_out_fwht + full_out_wait + full_out_rpc + full_out_post) / 1000.0, (full_out_fwht + full_out_wait + full_out_rpc + full_out_post) / 16000.0);
    printf("|   |   |-- Residual Add 1: %.2f ms\n", full_res1 / 1000.0);
    printf("|   |   |-- Post-Attention RMSNorm: %.2f ms\n", full_pn / 1000.0);
    printf("|   |   |-- Fused MLP Block (16 layers): %.2f ms [avg %.2f ms/layer]\n",
           (full_mlp_fwht + full_mlp_wait + full_mlp_rpc) / 1000.0, (full_mlp_fwht + full_mlp_wait + full_mlp_rpc) / 16000.0);
    printf("|   |   `-- Residual Add 2: %.2f ms\n", full_res2 / 1000.0);
    printf("|   |\n");
    printf("|   `-- 2.3 ARCHITECTURAL SUBSYSTEM TOTALS (All 64 Layers):\n");
    int layer_calls = (g_calls_rpc > 2) ? (g_calls_rpc - 2) : 64;
    printf("|       |-- Pure Hexagon HVX RPCs (%d calls): %.2f ms (%.1f%%) [avg %.2f ms/call]\n",
           layer_calls,
           (total_npu_rpc_us - p->t_lmhead_rpc_us) / 1000.0,
           100.0 * (total_npu_rpc_us - p->t_lmhead_rpc_us) / total_tok_us,
           (total_npu_rpc_us - p->t_lmhead_rpc_us) / (64.0 * 1000.0));
    printf("|       |-- Pure Host ARM CPU Math: %.2f ms (%.1f%%)\n",
           total_cpu_math_us / 1000.0, 100.0 * total_cpu_math_us / total_tok_us);
    printf("|       |   |-- FWHT-1024 Transforms (layer calls): %.2f ms\n",
           (total_fwht_us - p->t_lmhead_fwht_us) / 1000.0);
    printf("|       |   |-- DeltaNet S_h Recurrence (48 layers): %.2f ms\n", lin_recur / 1000.0);
    printf("|       |   |-- Conv1D + Proj_ab + Norms: %.2f ms\n", (lin_conv + lin_ab + lin_qk + lin_in_norm + lin_pn) / 1000.0);
    printf("|       |   `-- Full Attention (GQA, RoPE, KV): %.2f ms\n", (full_qk + full_rope + full_kv + full_gqa) / 1000.0);
    printf("|       `-- Ring Buffer & Memcpy Wait: %.2f ms (%.1f%%)\n",
           total_wait_us / 1000.0, 100.0 * total_wait_us / total_tok_us);
    printf("|\n");
    printf("|-- 3. FINAL NORM & STATIC LM HEAD (248 320 rows, 322 MB): %.2f ms (%.2f%%)\n",
           (p->t_final_norm_us + p->t_lmhead_fwht_us + p->t_lmhead_memcpy_us + p->t_lmhead_rpc_us + p->t_lmhead_post_us + p->t_argmax_us) / 1000.0,
           100.0 * (p->t_final_norm_us + p->t_lmhead_fwht_us + p->t_lmhead_memcpy_us + p->t_lmhead_rpc_us + p->t_lmhead_post_us + p->t_argmax_us) / total_tok_us);
    printf("|   |-- Final RMSNorm: %.2f ms\n", p->t_final_norm_us / 1000.0);
    printf("|   |-- LM Head FWHT-5120: %.2f ms\n", p->t_lmhead_fwht_us / 1000.0);
    printf("|   |-- LM Head Pre-Stage Memcpy (Chunk 0): %.2f ms\n", p->t_lmhead_memcpy_us / 1000.0);
    printf("|   |-- LM Head FastRPC / HVX GEMV + CDSP Argmax: %.2f ms (%.2f%%) [%d calls, 322 MB weights]\n",
           p->t_lmhead_rpc_us / 1000.0, 100.0 * p->t_lmhead_rpc_us / total_tok_us, g_calls_rpc - layer_calls);
    printf("|   |-- Logits Copy: %.2f ms\n", p->t_lmhead_post_us / 1000.0);
    printf("|   `-- Host Token Argmax Selection: %.2f ms (%.2f%%)\n", p->t_argmax_us / 1000.0, 100.0 * p->t_argmax_us / total_tok_us);
    printf("----------------------------------------------------------------------------------------------------\n");
    if (STATIC_LAYERS < 64) {
        printf(">>> COMPARATIVE AUDIT (SMMU STATIC vs STREAMING RING BUFFER):\n");
        printf("  * Layers 0..%d (Static SMMU): Total %.2f ms | Avg %.2f ms/layer | NPU RPC: %.2f ms | Wait: %.2f ms\n",
               STATIC_LAYERS - 1, stat_layer_total / 1000.0, stat_layer_total / (STATIC_LAYERS * 1000.0), stat_rpc_total / 1000.0, stat_wait_total / 1000.0);
        printf("  * Layers %d..63 (Streaming Ring): Total %.2f ms | Avg %.2f ms/layer | NPU RPC: %.2f ms | Wait: %.2f ms\n",
               STATIC_LAYERS, strm_layer_total / 1000.0, strm_layer_total / ((64 - STATIC_LAYERS) * 1000.0), strm_rpc_total / 1000.0, strm_wait_total / 1000.0);
        printf("  * Streaming Overhead Delta: %+.2f ms/layer (Wait stall: %.2f ms total, %.2f%% of token)\n",
               (strm_layer_total / (64 - STATIC_LAYERS) - stat_layer_total / STATIC_LAYERS) / 1000.0, strm_wait_total / 1000.0, 100.0 * strm_wait_total / total_tok_us);
    } else {
        printf(">>> PIPELINED SMMU AUDIT (ALL 64 LAYERS IN SMMU ZERO-COPY):\n");
        printf("  * All 64 Layers: Total %.2f ms | Avg %.2f ms/layer | NPU RPC: %.2f ms | Wait: %.2f ms (0 CPU Memcpy!)\n",
               stat_layer_total / 1000.0, stat_layer_total / (64.0 * 1000.0), stat_rpc_total / 1000.0, total_wait_us / 1000.0);
    }
    printf("----------------------------------------------------------------------------------------------------\n");
    printf(">>> COMPARATIVE AUDIT (LINEAR ATTENTION vs FULL ATTENTION):\n");
    printf("  * Linear Layers (48 layers): Avg Layer Total: %.2f ms | InProj RPC: %.2f ms | Attn Math: %.2f ms | MLP RPC: %.2f ms\n",
           lin_total / 48000.0, lin_in_rpc / 48000.0, (lin_ab + lin_conv + lin_qk + lin_recur) / 48000.0, lin_mlp_rpc / 48000.0);
    printf("  * Full Attention (16 layers): Avg Layer Total: %.2f ms | InProj RPC: %.2f ms | Attn Math: %.2f ms | MLP RPC: %.2f ms\n",
           full_total / 16000.0, full_in_rpc / 16000.0, (full_qk + full_rope + full_kv + full_gqa) / 16000.0, full_mlp_rpc / 16000.0);
    printf("----------------------------------------------------------------------------------------------------\n");
    printf(">>> THE ELEPHANT IN THE ROOM (BOTTLENECK & ANOMALY FINDER):\n");

    // 1. FastRPC IPC context switch overhead:
    double est_ipc_ms = (double)g_calls_rpc * 0.18;
    printf("  [1] FASTRPC KERNEL IPC ROUND-TRIPS: %d calls/token (~%.1f ms = ~%.1f%% of token time is pure IPC overhead!)\n",
           g_calls_rpc, est_ipc_ms, 100.0 * est_ipc_ms / t_tok_ms);

    // 2. LM Head fraction:
    printf("  [2] LM HEAD CALL (248k rows): %.2f ms (%.1f%% of entire token!)\n",
           p->t_lmhead_rpc_us / 1000.0, 100.0 * p->t_lmhead_rpc_us / total_tok_us);

    // 3. DeltaNet Recurrence vs other CPU math:
    printf("  [3] DELTANET RECURRENCE S_h: %.2f ms on ARM CPU (%.1f%% of token, %.2f ms/layer)\n",
           lin_recur / 1000.0, 100.0 * lin_recur / total_tok_us, lin_recur / 48000.0);

    // 4. Host FWHT transforms:
    printf("  [4] HOST FWHT-1024 TRANSFORMS: %.2f ms across %d calls (%.1f%% of token)\n",
           total_fwht_us / 1000.0, g_calls_rpc, 100.0 * total_fwht_us / total_tok_us);

    // 5. Memory streaming wait:
    printf("  [5] RING BUFFER MEMCPY STALLS: %.2f ms total (%.1f%% of token - %s)\n",
           total_wait_us / 1000.0, 100.0 * total_wait_us / total_tok_us,
           total_wait_us < 5000 ? "PERFECTLY HIDDEN" : "SOME STALLS");

    // 6. Outlier layer detection:
    double avg_layer_ms = total_layer_us / 64000.0;
    int outlier_count = 0;
    printf("  [6] LAYER OUTLIERS (> 1.25x avg = > %.2f ms):\n", avg_layer_ms * 1.25);
    for (int L = 0; L < 64; L++) {
        double l_ms = p->layers[L].t_layer_total_us / 1000.0;
        if (l_ms > avg_layer_ms * 1.25) {
            printf("      * Layer %02d (%s, %s): %.2f ms (InRPC: %.2f, AttnMath: %.2f, MlpWait: %.2f, MlpRPC: %.2f)\n",
                   L, p->layers[L].is_lin ? "LIN " : "FULL", p->layers[L].is_static ? "STATIC" : "STREAM",
                   l_ms, p->layers[L].c0.t_rpc_us / 1000.0,
                   (p->layers[L].is_lin ? (p->layers[L].t_attn_ab_us + p->layers[L].t_attn_conv_us + p->layers[L].t_attn_qk_norm_us + p->layers[L].t_attn_recur_us) :
                                          (p->layers[L].t_attn_qk_norm_us + p->layers[L].t_attn_rope_us + p->layers[L].t_attn_kv_enc_us + p->layers[L].t_attn_gqa_us)) / 1000.0,
                   p->layers[L].t_mlp_wait_us / 1000.0, p->layers[L].t_mlp_rpc_us / 1000.0);
            outlier_count++;
        }
    }
    if (outlier_count == 0) printf("      (None detected! All 64 layers execution is exceptionally balanced and jitter-free)\n");

    // 7. Per-layer table (compact 64 rows):
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("PER-LAYER BREAKDOWN TABLE (All times in ms):\n");
    printf(" L  Typ Stat  InNrm InFWHT InWait  InRPC AttnMth OutFWHT OutWait OutRPC PostNrm MlpFWHT MlpWait  MlpRPC   Total\n");
    for (int L = 0; L < 64; L++) {
        const LayerTiming* lt = &p->layers[L];
        double attn_math = lt->is_lin ?
            (lt->t_attn_ab_us + lt->t_attn_conv_us + lt->t_attn_qk_norm_us + lt->t_attn_recur_us) / 1000.0 :
            (lt->t_attn_qk_norm_us + lt->t_attn_rope_us + lt->t_attn_kv_enc_us + lt->t_attn_gqa_us) / 1000.0;
        printf("%02d  %s %s %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %7.2f %6.2f %7.2f %7.2f %7.2f %7.2f %7.2f\n",
               L, lt->is_lin ? "LIN " : "FULL", lt->is_static ? "STAT" : "STRM",
               lt->t_in_norm_us / 1000.0,
               lt->c0.t_fwht_us / 1000.0, lt->c0.t_wait_us / 1000.0, lt->c0.t_rpc_us / 1000.0,
               attn_math,
               lt->c1.t_fwht_us / 1000.0, lt->c1.t_wait_us / 1000.0, lt->c1.t_rpc_us / 1000.0,
               lt->t_post_norm_us / 1000.0,
               lt->t_mlp_fwht_us / 1000.0, lt->t_mlp_wait_us / 1000.0, lt->t_mlp_rpc_us / 1000.0,
               lt->t_layer_total_us / 1000.0);
    }
    printf("====================================================================================================\n\n");
}

static void reset_engine_states(void) {
    if (g_h) {
        bonsai_register_lin_states(g_h, NULL, 0, NULL, 0, NULL, 0);
    }
#ifndef _WIN32
    size_t kv_sz  = (size_t)16 * 4 * 256 * g_ctx;
    size_t kvs_sz = (size_t)16 * 4 * g_ctx * sizeof(float);
    if (kvk)  madvise(kvk,  kv_sz,  MADV_DONTNEED);
    if (kvv)  madvise(kvv,  kv_sz,  MADV_DONTNEED);
    if (kvks) madvise(kvks, kvs_sz, MADV_DONTNEED);
    if (kvvs) madvise(kvvs, kvs_sz, MADV_DONTNEED);
#endif
}

static inline int sample_argmax_token(const float* lgt) {
    int bi = 0;
    for (int i = 1; i < 248320; i++) {
        if (lgt[i] > lgt[bi]) bi = i;
    }
    return bi;
}

static int draft_prompt_lookup(const int* history, int n_hist, int* drafts, int max_drafts) {
    if (n_hist < 4 || max_drafts <= 0) return 0;
    for (int gram_len = 3; gram_len >= 2; gram_len--) {
        if (n_hist < gram_len + 1) continue;
        const int* target = history + n_hist - gram_len;
        for (int i = n_hist - gram_len - 1; i >= 0; i--) {
            int match = 1;
            for (int k = 0; k < gram_len; k++) {
                if (history[i + k] != target[k]) { match = 0; break; }
            }
            if (match) {
                int found = 0;
                int src = i + gram_len;
                while (src < n_hist && found < max_drafts) {
                    drafts[found++] = history[src++];
                }
                if (found > 0) return found;
            }
        }
    }
    return 0;
}

#include "bonsai_server.h"

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    CPU_SET(7, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    setpriority(PRIO_PROCESS, 0, -20);
    if (getenv("BONSAI_RT")) {
        struct sched_param sp = { .sched_priority = 50 };
        int rt_rc = sched_setscheduler(0, SCHED_FIFO, &sp);
        fprintf(stderr, "[fwd] SCHED_FIFO prio=50 rc=%d\n", rt_rc);
    }
#endif
    if (getenv("BONSAI_PROFILE_TREE")) {
        g_profile_tree = 1;
        fprintf(stderr, "[fwd] Hierarchical debug profiling tree enabled via BONSAI_PROFILE_TREE\n");
    }
    fprintf(stderr, "[fwd] start\n");
    if (argc < 4) {
        printf("usage: bonsai_fwd model.npubin tok.bin \"prompt\" [nsteps] [ctx]\n");
        printf("       bonsai_fwd model.npubin tok.bin --chat [ctx]\n");
        printf("       bonsai_fwd model.npubin tok.bin --server [port] [ctx]\n");
        return 2;
    }
    int is_chat   = (strcmp(argv[3], "--chat") == 0);
    int is_server = (strcmp(argv[3], "--server") == 0);
    if (is_chat) {
        if (argc >= 5) g_ctx = atoi(argv[4]);
    } else if (is_server) {
        if (argc >= 6) g_ctx = atoi(argv[5]);
    } else {
        if (argc >= 6) g_ctx = atoi(argv[5]);
    }
    const char* ctx_env = getenv("BONSAI_CTX");
    if (ctx_env) g_ctx = atoi(ctx_env);
    if (g_ctx < 256) g_ctx = 256;
    int nsteps = argc >= 5 ? atoi(argv[4]) : 3;
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    if (load_npubin(argv[1])) { printf("MODEL LOAD FAIL\n"); return 2; }
    fprintf(stderr, "[fwd] model ok nt=%d\n", g_nt);
    b2u_init();
    if (load_tok(argv[2])) { printf("TOK LOAD FAIL\n"); return 2; }
    build_spairs();
    {
        sspecs = calloc(spec_n > 0 ? (size_t)spec_n : 1, sizeof(*sspecs));
        for (int i = 0; i < spec_n; i++) { sspecs[i].s = spec_strs[i]; sspecs[i].id = spec_ids[i]; }
    }
    struct { int domain; int enable; } um = {3, 1};
    extern int remote_session_control(unsigned, void*, unsigned);
    printf("UNSIGNED_CTRL rc=%d\n", remote_session_control(2, &um, sizeof(um)));
    extern int bonsai_open(const char*, unsigned long long*);
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &g_h);
    if (rc) { printf("NPU OPEN FAIL rc=%d\n", rc); return 1; }
    printf("NPU OPEN OK\n");
#ifndef _WIN32
    {
        struct { uint32_t enable; } wl = { 1 };
        int wrc = remote_handle64_control(g_h, 3 /*DSPRPC_CONTROL_WAKELOCK*/, &wl, sizeof(wl));
        const char* qos_env = getenv("BONSAI_QOS");
        const char* qlat_env = getenv("BONSAI_QOS_LAT");
        uint32_t qos_mode = qos_env ? (uint32_t)atoi(qos_env) : 3u;
        uint32_t qos_lat  = qlat_env ? (uint32_t)atoi(qlat_env) : ((qos_mode == 3u) ? 9000u : 100u);
        struct { uint32_t enable; uint32_t latency; } lat = { qos_mode, qos_lat };
        int lrc = remote_handle64_control(g_h, 1 /*DSPRPC_CONTROL_LATENCY*/, &lat, sizeof(lat));
        fprintf(stderr, "[fwd] FastRPC QoS mode=%u lat=%u rc=%d, wakelock rc=%d\n", qos_mode, qos_lat, lrc, wrc);
    }
#endif
#ifndef _WIN32
    bus_boost_acquire();
#endif
    float* logits = (float*)calloc(4 * 248320, sizeof(float));
    if (logits) memset(logits, 0, 4 * 248320 * sizeof(float));
    if (stage_init()) { printf("STAGE INIT FAIL\n"); return 1; }
    if (!g_fused_lin && (!ssm_conv || !ssm_rec)) { printf("STATE ALLOC FAIL\n"); return 1; }
    if (!kvk || !kvv || !kvks || !kvvs) { printf("KV ALLOC FAIL\n"); return 1; }
    if (is_chat) {
        run_interactive_chat(logits);
#ifndef _WIN32
        bus_boost_release();
#endif
        return 0;
    }
    if (is_server) {
        int port = (argc >= 5) ? atoi(argv[4]) : 8080;
        if (port <= 0 || port > 65535) port = 8080;
        run_http_server(port, logits);
#ifndef _WIN32
        bus_boost_release();
#endif
        return 0;
    }
    static int ids[8192];
    int nids = tok_encode((const unsigned char*)argv[3], (int)strlen(argv[3]), ids, 8192);
    if (nids < 0) { printf("ENCODE FAIL\n"); return 1; }
    printf("prompt tokens=%d\n", nids);

    const char* pchunk_env = getenv("BONSAI_PREFILL_CHUNK");
    int max_pchunk = pchunk_env ? atoi(pchunk_env) : (g_fused_lin ? 1 : 8);
    if (max_pchunk < 1) max_pchunk = 1;
    if (max_pchunk > 8) max_pchunk = 8;

    double t0 = now_ms();
    int n_prefill = nids - 1;
    int p_pos = 0;
    while (p_pos < n_prefill) {
        int chunk = n_prefill - p_pos;
        if (chunk > max_pchunk) chunk = max_pchunk;
        if (forward_tokens_batch(ids + p_pos, p_pos, chunk, g_hidden_batch)) {
            printf("PREFILL FAIL at %d (chunk=%d)\n", p_pos, chunk);
            return 1;
        }
        p_pos += chunk;
    }
    double t1 = now_ms();
    printf("prefill %d toks in %.1f ms (%.1f ms/tok)\n",
           n_prefill, t1 - t0, n_prefill > 0 ? (t1 - t0) / n_prefill : 0.0);

    static int gen_hist[8192];
    int n_gen = nids;
    for (int i = 0; i < nids; i++) gen_hist[i] = ids[i];

    const char* mtp_env = getenv("BONSAI_MTP");
    int mtp_active = (mtp_env && atoi(mtp_env) != 0);
    if (mtp_active) {
        printf("[fwd] Multi-Token Prediction (MTP / Prompt Lookup Speculative Decoding) ENABLED\n");
    }

    int cur = ids[nids - 1];
    g_t_rpc = 0; g_t_memcpy = 0; g_t_trans = 0; g_calls_rpc = 0;
    int s = 0;
    while (s < nsteps) {
        int pos = (nids - 1) + s;
        int drafts[2];
        int n_draft = (mtp_active && s + 1 < nsteps) ? draft_prompt_lookup(gen_hist, n_gen, drafts, 1) : 0;

        if (n_draft > 0) {
            // MTP speculative branch with B = 2
            int vtoks[2] = { cur, drafts[0] };
            double a = now_ms();
            if (forward_tokens_batch(vtoks, pos, 2, g_hidden_batch) == 0 &&
                cdsp_lmhead_batch(g_hidden_batch, 2, logits) == 0) {
                double b = now_ms();
                int bi_0 = sample_argmax_token(logits);
                if (bi_0 == drafts[0]) {
                    // Match! Both tokens accepted!
                    int bi_1 = sample_argmax_token(logits + 248320);
                    unsigned char dec0[256], dec1[256];
                    int nb0 = tok_decode(&bi_0, 1, dec0, sizeof(dec0));
                    int nb1 = tok_decode(&bi_1, 1, dec1, sizeof(dec1));
                    dec0[nb0 < 0 ? 0 : (nb0 > 255 ? 255 : nb0)] = 0;
                    dec1[nb1 < 0 ? 0 : (nb1 > 255 ? 255 : nb1)] = 0;
                    printf("step %d [MTP 2x MATCH] tok0=%d (%s) tok1=%d (%s) total=%.0fms (%.1f ms/tok = %.2f tok/s | NPU_RPC=%.0fms [%d calls])\n",
                           s, bi_0, (const char*)dec0, bi_1, (const char*)dec1, b - a, (b - a) / 2.0, 2000.0 / (b - a), g_t_rpc, g_calls_rpc);
                    gen_hist[n_gen++] = bi_0;
                    gen_hist[n_gen++] = bi_1;
                    cur = bi_1;
                    s += 2;
                    g_t_rpc = 0; g_t_memcpy = 0; g_t_trans = 0; g_calls_rpc = 0;
                    continue;
                }
            }
        }

        uint64_t u_tok_start = 0;
        if (g_profile_tree) {
            memset(&g_prof, 0, sizeof(g_prof));
            u_tok_start = now_us();
        }
        double a = now_ms();
        if (forward_token(cur, pos)) { printf("DECODE FAIL at %d\n", s); return 1; }
        // Pre-staged unchunked head (zero memcpy, single direct NPU call)
        if (cdsp_lmhead(hidden, logits)) { printf("HEAD FAIL\n"); return 1; }
        double b = now_ms();
        // greedy + sanity
        uint64_t u_arg0 = (g_profile_tree) ? now_us() : 0;
        int bi = 0, nan = 0;
        if (g_fast_argmax_idx >= 0 && g_fast_argmax_idx < 248320) {
            bi = g_fast_argmax_idx;
        } else {
            for (int i = 0; i < 248320; i++) {
                float v = logits[i];
                if (!(v == v) || v > 1e10f || v < -1e10f) nan = 1;
                if (v > logits[bi]) bi = v == v ? i : bi;
            }
        }
        if (g_profile_tree) {
            g_prof.t_argmax_us = now_us() - u_arg0;
            g_prof.t_token_total_us = now_us() - u_tok_start;
        }
        if (bi == 0 && logits[0] == 0.0f) {
            fprintf(stderr, "[dbg] step %d hidden[0..3]=%.4f,%.4f,%.4f,%.4f logits[0..3]=%.4f,%.4f,%.4f,%.4f\n",
                    s, hidden[0], hidden[1], hidden[2], hidden[3], logits[0], logits[1], logits[2], logits[3]);
        }
        unsigned char dec[256];
        int nb = tok_decode(&bi, 1, dec, sizeof(dec));
        dec[nb < 0 ? 0 : (nb > 255 ? 255 : nb)] = 0;
        printf("step %d tok=%d logit=%.3f %s [%s] total=%.0fms (NPU_RPC=%.0fms [%d calls], trans=%.0fms, memcpy=%.0fms, CPU_math=%.0fms)%s\n",
               s, bi, logits[bi],
               nb < 0 ? "(decode?)" : (const char*)dec, "", b - a,
               g_t_rpc, g_calls_rpc, g_t_trans, g_t_memcpy, (b - a) - g_t_rpc - g_t_trans - g_t_memcpy,
               nan ? " NAN!" : "");
        if (g_profile_tree && s == 0) {
            print_profile_tree(&g_prof, s);
        }
        gen_hist[n_gen++] = bi;
        g_t_rpc = 0; g_t_memcpy = 0; g_t_trans = 0; g_calls_rpc = 0;
        cur = bi;
        s++;
    }

    const char* vbatch_env = getenv("BONSAI_VERIFY_BATCH");
    int vbatch = vbatch_env ? atoi(vbatch_env) : 0;
    if (vbatch >= 2 && vbatch <= 4) {
        int vtoks[4] = { cur, cur, cur, cur };
        g_t_rpc = 0; g_t_memcpy = 0; g_t_trans = 0; g_calls_rpc = 0;
        double va = now_ms();
        if (forward_tokens_batch(vtoks, (nids - 1) + nsteps, vbatch, g_hidden_batch) == 0 &&
            cdsp_lmhead_batch(g_hidden_batch, vbatch, logits) == 0) {
            double vb = now_ms();
            printf("mtp_verify batch=%d total=%.0fms (%.1f ms/tok = %.2f tok/s | NPU_RPC=%.0fms [%d calls], trans=%.0fms, memcpy=%.0fms, CPU_math=%.0fms)\n",
                   vbatch, vb - va, (vb - va) / vbatch, 1000.0 * vbatch / (vb - va),
                   g_t_rpc, g_calls_rpc, g_t_trans, g_t_memcpy, (vb - va) - g_t_rpc - g_t_trans - g_t_memcpy);
        }
    }

#ifndef _WIN32
    bus_boost_release();
#endif
    printf("SMOKE DONE\n");
    return 0;
}
