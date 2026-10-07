// Ultra-optimized HVX Q1_0 (Binary) & Q2_0 (Interleaved Ternary) GEMV/GEMM kernel
// for Hexagon v79 (1024-bit SIMD).
// Multi-threaded across 6 HVX threads via QuRT with:
//   - Zero-spill scale-factored qf32 accumulation
//   - Direct 2-instruction bitmask testing (vand + vcmp_eq)
//   - Branchless 3-instruction FP16->FP32 scale conversion
//   - Automatic Binary Q1 vs Interleaved Ternary Q2 dispatch via prow

#include <string.h>
#include <stdint.h>
#include <math.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#ifndef SIM_NO_QURT
#include "qurt.h"
#endif

// Branchless FP16 -> FP32 scale * 0.5f (for Binary Q1 where stored scale is 2*s)
static inline unsigned f16_scale_to_f32_bits_half(unsigned short h) {
    unsigned f = ((unsigned)h << 13) + 0x37800000u;
    return (h < 0x0400) ? 0 : f;
}

// Branchless FP16 -> FP32 full scale s (for Ternary Q2 where stored scale is s)
static inline unsigned f16_scale_to_f32_bits_full(unsigned short h) {
    unsigned f = ((unsigned)h << 13) + 0x38000000u;
    return (h < 0x0400) ? 0 : f;
}

static inline int hvx_vextract(HVX_Vector v, int off) {
    int res;
    asm("%0 = vextract(%1,%2)" : "=r"(res) : "v"(v), "r"(off));
    return res;
}

static inline float hvx_reduce_sum32(HVX_Vector v) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
    int bits = hvx_vextract(v, 0);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

#define HVX_MAX_K 17408
#define HVX_MAX_NG (HVX_MAX_K >> 7)
#define HVX_MAX_BATCH 32
static float hvx_xalign[HVX_MAX_BATCH * HVX_MAX_K] __attribute__((aligned(128)));
static int8_t hvx_qx8[HVX_MAX_BATCH * HVX_MAX_K] __attribute__((aligned(128)));
static float hvx_sx[HVX_MAX_BATCH * HVX_MAX_NG] __attribute__((aligned(128)));
static float hvx_sx4[HVX_MAX_BATCH * HVX_MAX_NG * 4] __attribute__((aligned(128)));

static const uint32_t hvx_bitmask_arr[32] __attribute__((aligned(128))) = {
    1u<<0,  1u<<1,  1u<<2,  1u<<3,  1u<<4,  1u<<5,  1u<<6,  1u<<7,
    1u<<8,  1u<<9,  1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
    1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
    1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31
};

static void quantize_x_q8_permuted(const float* x, int batch, int in_dim) {
    int ng = in_dim >> 7;
    HVX_Vector vBias   = Q6_V_vsplat_R((int)0x4B400000u); // 12582912.0f magic rounding bias
    HVX_Vector vMaskFF = Q6_V_vsplat_R(0x000000FF);
    HVX_Vector vMinQ   = Q6_V_vsplat_R(-127);
    HVX_Vector vMaxQ   = Q6_V_vsplat_R(127);
    uint32_t pbuf[4][32] __attribute__((aligned(128)));

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        const float* xb = x + (unsigned)b_idx * (unsigned)in_dim;
        int8_t* qdst_b = hvx_qx8 + (unsigned)b_idx * (unsigned)in_dim;
        float* sdst_b = hvx_sx + (unsigned)b_idx * (unsigned)ng;
        float* sdst4_b = hvx_sx4 + (unsigned)b_idx * (unsigned)(ng << 2);
        for (int g8 = 0; g8 < ng; g8 += 8) {
            int8_t* qdst_super = qdst_b + (unsigned)g8 * 128u;
            for (int sub = 0; sub < 8; sub++) {
                int g = g8 + sub;
                const HVX_Vector* vgx = (const HVX_Vector*)(xb + (unsigned)g * 128u);
                HVX_Vector vA0 = Q6_Vsf_vabs_Vsf(vgx[0]);
                HVX_Vector vA1 = Q6_Vsf_vabs_Vsf(vgx[1]);
                HVX_Vector vA2 = Q6_Vsf_vabs_Vsf(vgx[2]);
                HVX_Vector vA3 = Q6_Vsf_vabs_Vsf(vgx[3]);
                HVX_Vector vMax = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmax_VsfVsf(vA0, vA1), Q6_Vsf_vmax_VsfVsf(vA2, vA3));
                vMax = Q6_Vsf_vmax_VsfVsf(vMax, Q6_V_vror_VR(vMax, 64));
                vMax = Q6_Vsf_vmax_VsfVsf(vMax, Q6_V_vror_VR(vMax, 32));
                vMax = Q6_Vsf_vmax_VsfVsf(vMax, Q6_V_vror_VR(vMax, 16));
                vMax = Q6_Vsf_vmax_VsfVsf(vMax, Q6_V_vror_VR(vMax, 8));
                vMax = Q6_Vsf_vmax_VsfVsf(vMax, Q6_V_vror_VR(vMax, 4));
                int amax_bits = hvx_vextract(vMax, 0);
                float amax;
                memcpy(&amax, &amax_bits, 4);

                float sx = amax * (1.0f / 127.0f);
                float inv_sx = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                sdst_b[g] = sx;
                sdst4_b[(g << 2) + 0] = sx;
                sdst4_b[(g << 2) + 1] = sx;
                sdst4_b[(g << 2) + 2] = sx;
                sdst4_b[(g << 2) + 3] = sx;

                int32_t inv_bits;
                memcpy(&inv_bits, &inv_sx, 4);
                HVX_Vector vInvSx = Q6_V_vsplat_R(inv_bits);

                for (int m = 0; m < 4; m++) {
                    HVX_Vector vScaled = Q6_Vsf_vmpy_VsfVsf(vgx[m], vInvSx);
                    HVX_Vector vBiased = Q6_Vsf_vadd_VsfVsf(vScaled, vBias);
                    HVX_Vector vQm     = Q6_Vw_vsub_VwVw(vBiased, vBias);
                    vQm = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(vQm, vMaxQ), vMinQ);
                    HVX_Vector vB0 = Q6_V_vand_VV(vQm, vMaskFF);
                    HVX_Vector vB1 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_V_vror_VR(vQm, 32), vMaskFF), 8);
                    HVX_Vector vB2 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_V_vror_VR(vQm, 64), vMaskFF), 16);
                    HVX_Vector vB3 = Q6_Vw_vasl_VwR(Q6_V_vror_VR(vQm, 96), 24);
                    *(HVX_Vector*)pbuf[m] = Q6_V_vor_VV(Q6_V_vor_VV(vB0, vB1), Q6_V_vor_VV(vB2, vB3));
                }
                for (int k = 0; k < 8; k++) {
                    uint32_t* dst32 = (uint32_t*)(qdst_super + (unsigned)k * 128u + (unsigned)sub * 16u);
                    dst32[0] = pbuf[0][k];
                    dst32[1] = pbuf[1][k];
                    dst32[2] = pbuf[2][k];
                    dst32[3] = pbuf[3][k];
                }
            }
        }
    }
}

#define NUM_WORKERS 6
#define STACK_SIZE (64 * 1024)

static char g_stack[NUM_WORKERS][STACK_SIZE] __attribute__((aligned(128)));
static qurt_thread_t g_tids[NUM_WORKERS];
static qurt_sem_t g_sem_start[NUM_WORKERS];
static qurt_sem_t g_sem_done[NUM_WORKERS];
static volatile int g_pool_ready = 0;
static volatile int g_pool_exit = 0;

typedef HVX_Vector HVX_UVector __attribute__((aligned(4)));

typedef enum {
    TASK_GEMV = 0,
    TASK_SWIGLU_FWHT = 1,
    TASK_DELTANET = 2,
    TASK_FWHT_BLOCK = 3,
    TASK_LIN_PREP = 4,
    TASK_FULL_GQA = 5
} TaskType;

typedef struct {
    TaskType task_type;
    int r0, r1;
    int batch, out_dim, in_dim, prow;
    const uint32_t* bits_u32;
    const unsigned short* scales;
    float* y;
    int blk_start, blk_end;
    const float* gate_up;
    const float* signs;
    float* out;

    // DeltaNet recurrence fields
    int h0, h1;
    const float* alog;
    const float* dtb;
    const float* avec;
    const float* bvec;
    const float* q;
    const float* k;
    const float* v;
    const float* z;
    const float* nw;
    float* S;
    float* no;

    // FWHT block fields
    float* fwht_data;
    const float* fwht_signs;

    // Linear prep fields (in_proj_a/b + Conv1D + SiLU)
    const float* x_in;
    const uint32_t* wa;
    const uint32_t* wb;
    float* avec_out;
    float* bvec_out;
    const float* cw;
    float* conv;
    float* qkv;
    float* sc;
    int c0, c1;
} WorkerTask;

static WorkerTask g_tasks[NUM_WORKERS];
#define MAX_FULL_CTX 4096
static float g_dsp_sc_buf[NUM_WORKERS][MAX_FULL_CTX] __attribute__((aligned(128)));
extern int g_dsp_kv_ctx_max;
extern int g_dsp_turbo4;

// ============================================================================
// 1. Single-Token Binary Q1 Slice (prow == in_dim / 8)
// ============================================================================
static void process_slice_q1(int r0, int r1, int in_dim, int prow,
                             const uint32_t* bits_u32, const unsigned short* scales,
                             float* y) {
    int ng = in_dim >> 7;
    int prow_u32 = prow >> 2;
    uint32_t l2_bits_cfg = 0x00808000u | (uint32_t)((4 * prow) >> 7);
    HVX_Vector vMask = *(const HVX_Vector*)hvx_bitmask_arr;

    int i = r0;
    for (; i + 3 < r1; i += 4) {
        if (i + 7 < r1) {
            Q6_l2fetch_AR((void*)(bits_u32 + (unsigned)(i + 4) * (unsigned)prow_u32), (int)l2_bits_cfg);
        }
        const unsigned short* sg0 = scales + (unsigned)(i + 0) * (unsigned)ng;
        const unsigned short* sg1 = scales + (unsigned)(i + 1) * (unsigned)ng;
        const unsigned short* sg2 = scales + (unsigned)(i + 2) * (unsigned)ng;
        const unsigned short* sg3 = scales + (unsigned)(i + 3) * (unsigned)ng;

        const uint32_t* bp0 = bits_u32 + (unsigned)(i + 0) * (unsigned)prow_u32;
        const uint32_t* bp1 = bits_u32 + (unsigned)(i + 1) * (unsigned)prow_u32;
        const uint32_t* bp2 = bits_u32 + (unsigned)(i + 2) * (unsigned)prow_u32;
        const uint32_t* bp3 = bits_u32 + (unsigned)(i + 3) * (unsigned)prow_u32;

        HVX_Vector vAcc0 = Q6_V_vzero();
        HVX_Vector vAcc1 = Q6_V_vzero();
        HVX_Vector vAcc2 = Q6_V_vzero();
        HVX_Vector vAcc3 = Q6_V_vzero();

        for (int g = 0; g < ng; g++) {
            const uint32_t* gb0 = bp0 + (unsigned)g * 4u;
            const uint32_t* gb1 = bp1 + (unsigned)g * 4u;
            const uint32_t* gb2 = bp2 + (unsigned)g * 4u;
            const uint32_t* gb3 = bp3 + (unsigned)g * 4u;
            const float* gx = hvx_xalign + (unsigned)g * 128u;

            HVX_Vector vGAcc0 = Q6_V_vzero();
            HVX_Vector vGAcc1 = Q6_V_vzero();
            HVX_Vector vGAcc2 = Q6_V_vzero();
            HVX_Vector vGAcc3 = Q6_V_vzero();

#pragma clang loop unroll(disable)
            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);
                HVX_Vector vNX = Q6_Vsf_vfneg_Vsf(vX);

                HVX_Vector vT0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb0[c]), vMask), vMask), vX, vNX);
                HVX_Vector vT1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb1[c]), vMask), vMask), vX, vNX);
                HVX_Vector vT2 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb2[c]), vMask), vMask), vX, vNX);
                HVX_Vector vT3 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb3[c]), vMask), vMask), vX, vNX);

                vGAcc0 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc0, vT0);
                vGAcc1 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc1, vT1);
                vGAcc2 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc2, vT2);
                vGAcc3 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc3, vT3);
            }

            unsigned sbits0 = f16_scale_to_f32_bits_half(sg0[g]);
            unsigned sbits1 = f16_scale_to_f32_bits_half(sg1[g]);
            unsigned sbits2 = f16_scale_to_f32_bits_half(sg2[g]);
            unsigned sbits3 = f16_scale_to_f32_bits_half(sg3[g]);

            vAcc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc0), Q6_V_vsplat_R((int)sbits0)));
            vAcc1 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc1), Q6_V_vsplat_R((int)sbits1)));
            vAcc2 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc2, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc2), Q6_V_vsplat_R((int)sbits2)));
            vAcc3 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc3, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc3), Q6_V_vsplat_R((int)sbits3)));
        }
        y[i + 0] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc0));
        y[i + 1] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc1));
        y[i + 2] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc2));
        y[i + 3] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc3));
    }

    for (; i < r1; i++) {
        const unsigned short* sg = scales + (unsigned)i * (unsigned)ng;
        const uint32_t* bp = bits_u32 + (unsigned)i * (unsigned)prow_u32;
        HVX_Vector vAcc = Q6_V_vzero();
        for (int g = 0; g < ng; g++) {
            const uint32_t* gb = bp + (unsigned)g * 4u;
            const float* gx = hvx_xalign + (unsigned)g * 128u;
            HVX_Vector vGAcc = Q6_V_vzero();
#pragma clang loop unroll(disable)
            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);
                HVX_Vector vNX = Q6_Vsf_vfneg_Vsf(vX);
                HVX_Vector vT = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb[c]), vMask), vMask), vX, vNX);
                vGAcc = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc, vT);
            }
            unsigned sbits = f16_scale_to_f32_bits_half(sg[g]);
            vAcc = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc), Q6_V_vsplat_R((int)sbits)));
        }
        y[i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc));
    }
}

// ============================================================================
// 2. Single-Token Ternary Q2 Slice (prow == 2 * (in_dim / 8))
//    Row layout in bonsai2-27b.npubin:
//      pos_bp = bp0 + 0              (half_prow_u32 = ng * 4 words for +1)
//      neg_bp = bp0 + half_prow_u32  (half_prow_u32 = ng * 4 words for -1)
// ============================================================================
static __attribute__((noinline)) void process_slice_q2_core(int r0, int r1, int in_dim, int prow,
                                  const uint32_t* bits_u32, const unsigned short* scales,
                                  const int8_t* qx8_base, const float* sx4_base,
                                  float* y) {
    int ng = in_dim >> 7;
    int prow_u32 = prow >> 2;
    int half_prow_u32 = prow_u32 >> 1;
    uint32_t l2_bits_cfg = 0x00808000u | (uint32_t)((4 * prow) >> 7);
    HVX_Vector vOnes = Q6_V_vsplat_R(0x01010101);

    int i = r0;
    for (; i + 3 < r1; i += 4) {
        if (i + 7 < r1) {
            Q6_l2fetch_AR((void*)(bits_u32 + (unsigned)(i + 4) * (unsigned)prow_u32), (int)l2_bits_cfg);
        }
        const unsigned short* sg0 = scales + (unsigned)(i + 0) * (unsigned)ng;
        const unsigned short* sg1 = scales + (unsigned)(i + 1) * (unsigned)ng;
        const unsigned short* sg2 = scales + (unsigned)(i + 2) * (unsigned)ng;
        const unsigned short* sg3 = scales + (unsigned)(i + 3) * (unsigned)ng;

        const uint32_t* p_bp0 = bits_u32 + (unsigned)(i + 0) * (unsigned)prow_u32;
        const uint32_t* n_bp0 = p_bp0 + (unsigned)half_prow_u32;
        const uint32_t* p_bp1 = bits_u32 + (unsigned)(i + 1) * (unsigned)prow_u32;
        const uint32_t* n_bp1 = p_bp1 + (unsigned)half_prow_u32;
        const uint32_t* p_bp2 = bits_u32 + (unsigned)(i + 2) * (unsigned)prow_u32;
        const uint32_t* n_bp2 = p_bp2 + (unsigned)half_prow_u32;
        const uint32_t* p_bp3 = bits_u32 + (unsigned)(i + 3) * (unsigned)prow_u32;
        const uint32_t* n_bp3 = p_bp3 + (unsigned)half_prow_u32;

        int one_r = 0x01010101;
        asm volatile("" : "+r"(one_r));
        HVX_Vector vOneTmp = Q6_V_vsplat_R(one_r);
        HVX_Vector vAcc4 = Q6_Vqf32_vsub_VsfVsf(vOneTmp, vOneTmp);

        for (int g = 0; g < ng; g += 8) {
            unsigned goff_u32 = (unsigned)g * 4u;
            HVX_Vector vP0 = *(const HVX_Vector*)(p_bp0 + goff_u32);
            HVX_Vector vN0 = *(const HVX_Vector*)(n_bp0 + goff_u32);
            HVX_Vector vP1 = *(const HVX_Vector*)(p_bp1 + goff_u32);
            HVX_Vector vN1 = *(const HVX_Vector*)(n_bp1 + goff_u32);
            HVX_Vector vP2 = *(const HVX_Vector*)(p_bp2 + goff_u32);
            HVX_Vector vN2 = *(const HVX_Vector*)(n_bp2 + goff_u32);
            HVX_Vector vP3 = *(const HVX_Vector*)(p_bp3 + goff_u32);
            HVX_Vector vN3 = *(const HVX_Vector*)(n_bp3 + goff_u32);

            const HVX_Vector* vX_vec = (const HVX_Vector*)(qx8_base + (unsigned)g * 128u);
            HVX_Vector vX0 = vX_vec[0];
            HVX_Vector vDot0 = Q6_Vw_vrmpy_VbVb(Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP0, vOnes), Q6_V_vand_VV(vN0, vOnes)), vX0);
            HVX_Vector vDot1 = Q6_Vw_vrmpy_VbVb(Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP1, vOnes), Q6_V_vand_VV(vN1, vOnes)), vX0);
            HVX_Vector vDot2 = Q6_Vw_vrmpy_VbVb(Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP2, vOnes), Q6_V_vand_VV(vN2, vOnes)), vX0);
            HVX_Vector vDot3 = Q6_Vw_vrmpy_VbVb(Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP3, vOnes), Q6_V_vand_VV(vN3, vOnes)), vX0);

#pragma clang loop unroll(full)
            for (int k = 1; k < 8; k++) {
                HVX_Vector vXk = vX_vec[k];
                vP0 = Q6_Vuw_vlsr_VuwR(vP0, 1); vN0 = Q6_Vuw_vlsr_VuwR(vN0, 1);
                vP1 = Q6_Vuw_vlsr_VuwR(vP1, 1); vN1 = Q6_Vuw_vlsr_VuwR(vN1, 1);
                vP2 = Q6_Vuw_vlsr_VuwR(vP2, 1); vN2 = Q6_Vuw_vlsr_VuwR(vN2, 1);
                vP3 = Q6_Vuw_vlsr_VuwR(vP3, 1); vN3 = Q6_Vuw_vlsr_VuwR(vN3, 1);
                vDot0 = Q6_Vw_vrmpyacc_VwVbVb(vDot0, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP0, vOnes), Q6_V_vand_VV(vN0, vOnes)), vXk);
                vDot1 = Q6_Vw_vrmpyacc_VwVbVb(vDot1, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP1, vOnes), Q6_V_vand_VV(vN1, vOnes)), vXk);
                vDot2 = Q6_Vw_vrmpyacc_VwVbVb(vDot2, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP2, vOnes), Q6_V_vand_VV(vN2, vOnes)), vXk);
                vDot3 = Q6_Vw_vrmpyacc_VwVbVb(vDot3, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP3, vOnes), Q6_V_vand_VV(vN3, vOnes)), vXk);
            }

            HVX_VectorPair W01   = Q6_W_vshuff_VVR(vDot1, vDot0, 4);
            HVX_Vector     v01   = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W01), Q6_V_hi_W(W01));
            HVX_VectorPair W23   = Q6_W_vshuff_VVR(vDot3, vDot2, 4);
            HVX_Vector     v23   = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W23), Q6_V_hi_W(W23));
            HVX_VectorPair W0123 = Q6_W_vshuff_VVR(v23, v01, 8);
            HVX_Vector     vDot4 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W0123), Q6_V_hi_W(W0123));

            asm volatile("" : "+v"(vDot4));

            uintptr_t a0 = (uintptr_t)(sg0 + g);
            uintptr_t a1 = (uintptr_t)(sg1 + g);
            uintptr_t a2 = (uintptr_t)(sg2 + g);
            uintptr_t a3 = (uintptr_t)(sg3 + g);

            HVX_Vector vS0 = Q6_V_vror_VR(*(const HVX_Vector*)(a0 & ~(uintptr_t)127), (int)(a0 & 127));
            HVX_Vector vS1 = Q6_V_vror_VR(*(const HVX_Vector*)(a1 & ~(uintptr_t)127), (int)(a1 & 127));
            HVX_Vector vS2 = Q6_V_vror_VR(*(const HVX_Vector*)(a2 & ~(uintptr_t)127), (int)(a2 & 127));
            HVX_Vector vS3 = Q6_V_vror_VR(*(const HVX_Vector*)(a3 & ~(uintptr_t)127), (int)(a3 & 127));

            HVX_Vector vS01   = Q6_V_lo_W(Q6_W_vshuff_VVR(vS1, vS0, -2));
            HVX_Vector vS23   = Q6_V_lo_W(Q6_W_vshuff_VVR(vS3, vS2, -2));
            HVX_Vector vS0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(vS23, vS01, -4));
            HVX_Vector vH     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(vS0123));

            int bias38_r = 0x38000000;
            asm volatile("" : "+r"(bias38_r));
            HVX_Vector vSW4    = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(vH, 13), Q6_V_vsplat_R(bias38_r));
            HVX_Vector vSX4    = *(const HVX_Vector*)(sx4_base + (unsigned)g * 4u);
            HVX_Vector vDotSX  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(vDot4), vSX4));

            vAcc4 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc4, Q6_Vqf32_vmpy_VsfVsf(vDotSX, vSW4));
        }
        HVX_Vector vSum = Q6_Vsf_equals_Vqf32(vAcc4);
        vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 64));
        vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 32));
        vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 16));
        int b0 = hvx_vextract(vSum, 0);
        int b1 = hvx_vextract(vSum, 4);
        int b2 = hvx_vextract(vSum, 8);
        int b3 = hvx_vextract(vSum, 12);
        memcpy(&y[i + 0], &b0, 4);
        memcpy(&y[i + 1], &b1, 4);
        memcpy(&y[i + 2], &b2, 4);
        memcpy(&y[i + 3], &b3, 4);
    }

    for (; i < r1; i++) {
        const unsigned short* sg = scales + (unsigned)i * (unsigned)ng;
        const uint32_t* p_bp = bits_u32 + (unsigned)i * (unsigned)prow_u32;
        const uint32_t* n_bp = p_bp + (unsigned)half_prow_u32;
        int one_r = 0x01010101;
        asm volatile("" : "+r"(one_r));
        HVX_Vector vOneTmp = Q6_V_vsplat_R(one_r);
        HVX_Vector vAcc = Q6_Vqf32_vsub_VsfVsf(vOneTmp, vOneTmp);
        for (int g = 0; g < ng; g += 8) {
            unsigned goff_u32 = (unsigned)g * 4u;
            HVX_Vector vP = *(const HVX_Vector*)(p_bp + goff_u32);
            HVX_Vector vN = *(const HVX_Vector*)(n_bp + goff_u32);
            const HVX_Vector* vX_vec = (const HVX_Vector*)(qx8_base + (unsigned)g * 128u);
            HVX_Vector vDot = Q6_Vw_vrmpy_VbVb(Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP, vOnes), Q6_V_vand_VV(vN, vOnes)), vX_vec[0]);
            for (int k = 1; k < 8; k++) {
                vP = Q6_Vuw_vlsr_VuwR(vP, 1);
                vN = Q6_Vuw_vlsr_VuwR(vN, 1);
                vDot = Q6_Vw_vrmpyacc_VwVbVb(vDot, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP, vOnes), Q6_V_vand_VV(vN, vOnes)), vX_vec[k]);
            }
            asm volatile("" : "+v"(vDot));
            uintptr_t a0 = (uintptr_t)(sg + g);
            HVX_Vector vS0 = Q6_V_vror_VR(*(const HVX_Vector*)(a0 & ~(uintptr_t)127), (int)(a0 & 127));
            HVX_Vector vU0 = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(vS0))));
            HVX_Vector vH  = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(vU0));
            vH = Q6_Vw_vadd_VwVw(vH, Q6_V_vror_VR(vH, 124));
            vH = Q6_Vw_vadd_VwVw(vH, Q6_V_vror_VR(vH, 120));
            int bias38_r = 0x38000000;
            asm volatile("" : "+r"(bias38_r));
            HVX_Vector vSW = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(vH, 13), Q6_V_vsplat_R(bias38_r));
            HVX_Vector vSX4 = *(const HVX_Vector*)(sx4_base + (unsigned)g * 4u);
            HVX_Vector vDotSX = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(vDot), vSX4));
            vAcc = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc, Q6_Vqf32_vmpy_VsfVsf(vDotSX, vSW));
        }
        y[i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc));
    }
}

static void process_slice_q2(int r0, int r1, int in_dim, int prow,
                             const uint32_t* bits_u32, const unsigned short* scales,
                             float* y) {
    process_slice_q2_core(r0, r1, in_dim, prow, bits_u32, scales, hvx_qx8, hvx_sx4, y);
}

// ============================================================================
// 3. Batched Binary Q1 Slice (batch > 1) - Zero-Spill 8-Accumulator Tiling
// ============================================================================
static void process_slice_batch_q1(int r0, int r1, int batch, int out_dim, int in_dim, int prow,
                                   const uint32_t* bits_u32, const unsigned short* scales,
                                   float* y) {
    int ng = in_dim >> 7;
    int prow_u32 = prow >> 2;
    HVX_Vector vMask = *(const HVX_Vector*)hvx_bitmask_arr;

    for (int r_base = r0; r_base < r1; r_base += 32) {
        int r_end = (r_base + 32 < r1) ? (r_base + 32) : r1;
        int b0 = 0;
        for (; b0 + 3 < batch; b0 += 4) {
            const float* x_b0 = hvx_xalign + (unsigned)(b0 + 0) * (unsigned)in_dim;
            const float* x_b1 = hvx_xalign + (unsigned)(b0 + 1) * (unsigned)in_dim;
            const float* x_b2 = hvx_xalign + (unsigned)(b0 + 2) * (unsigned)in_dim;
            const float* x_b3 = hvx_xalign + (unsigned)(b0 + 3) * (unsigned)in_dim;

            int i = r_base;
            for (; i + 1 < r_end; i += 2) {
                const unsigned short* sg0 = scales + (unsigned)(i + 0) * (unsigned)ng;
                const unsigned short* sg1 = scales + (unsigned)(i + 1) * (unsigned)ng;
                const uint32_t* bp0 = bits_u32 + (unsigned)(i + 0) * (unsigned)prow_u32;
                const uint32_t* bp1 = bits_u32 + (unsigned)(i + 1) * (unsigned)prow_u32;

                HVX_Vector vAcc00 = Q6_V_vzero(), vAcc01 = Q6_V_vzero(), vAcc02 = Q6_V_vzero(), vAcc03 = Q6_V_vzero();
                HVX_Vector vAcc10 = Q6_V_vzero(), vAcc11 = Q6_V_vzero(), vAcc12 = Q6_V_vzero(), vAcc13 = Q6_V_vzero();

                for (int g = 0; g < ng; g++) {
                    unsigned sbits0 = f16_scale_to_f32_bits_half(sg0[g]);
                    unsigned sbits1 = f16_scale_to_f32_bits_half(sg1[g]);
                    HVX_Vector vS0 = Q6_V_vsplat_R((int)sbits0), vNS0 = Q6_V_vsplat_R((int)(sbits0 ^ 0x80000000u));
                    HVX_Vector vS1 = Q6_V_vsplat_R((int)sbits1), vNS1 = Q6_V_vsplat_R((int)(sbits1 ^ 0x80000000u));

                    const uint32_t* gb0 = bp0 + (unsigned)g * 4u;
                    const uint32_t* gb1 = bp1 + (unsigned)g * 4u;
                    unsigned goff = (unsigned)g * 128u;

        #pragma clang loop unroll(disable)
            for (int c = 0; c < 4; c++) {
                        unsigned coff = goff + (unsigned)c * 32u;
                        HVX_Vector vWgt0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb0[c]), vMask), vMask), vS0, vNS0);
                        HVX_Vector vWgt1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb1[c]), vMask), vMask), vS1, vNS1);

                        HVX_Vector vX0 = *(const HVX_Vector*)(x_b0 + coff);
                        vAcc00 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc00, Q6_Vqf32_vmpy_VsfVsf(vWgt0, vX0));
                        vAcc10 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc10, Q6_Vqf32_vmpy_VsfVsf(vWgt1, vX0));

                        HVX_Vector vX1 = *(const HVX_Vector*)(x_b1 + coff);
                        vAcc01 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc01, Q6_Vqf32_vmpy_VsfVsf(vWgt0, vX1));
                        vAcc11 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc11, Q6_Vqf32_vmpy_VsfVsf(vWgt1, vX1));

                        HVX_Vector vX2 = *(const HVX_Vector*)(x_b2 + coff);
                        vAcc02 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc02, Q6_Vqf32_vmpy_VsfVsf(vWgt0, vX2));
                        vAcc12 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc12, Q6_Vqf32_vmpy_VsfVsf(vWgt1, vX2));

                        HVX_Vector vX3 = *(const HVX_Vector*)(x_b3 + coff);
                        vAcc03 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc03, Q6_Vqf32_vmpy_VsfVsf(vWgt0, vX3));
                        vAcc13 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc13, Q6_Vqf32_vmpy_VsfVsf(vWgt1, vX3));
                    }
                }
                y[(b0 + 0) * out_dim + i + 0] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc00));
                y[(b0 + 0) * out_dim + i + 1] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc10));
                y[(b0 + 1) * out_dim + i + 0] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc01));
                y[(b0 + 1) * out_dim + i + 1] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc11));
                y[(b0 + 2) * out_dim + i + 0] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc02));
                y[(b0 + 2) * out_dim + i + 1] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc12));
                y[(b0 + 3) * out_dim + i + 0] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc03));
                y[(b0 + 3) * out_dim + i + 1] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc13));
            }
            for (; i < r_end; i++) {
                const unsigned short* sg = scales + (unsigned)i * (unsigned)ng;
                const uint32_t* bp = bits_u32 + (unsigned)i * (unsigned)prow_u32;
                HVX_Vector vAcc0 = Q6_V_vzero(), vAcc1 = Q6_V_vzero(), vAcc2 = Q6_V_vzero(), vAcc3 = Q6_V_vzero();
                for (int g = 0; g < ng; g++) {
                    unsigned sbits = f16_scale_to_f32_bits_half(sg[g]);
                    HVX_Vector vS = Q6_V_vsplat_R((int)sbits), vNS = Q6_V_vsplat_R((int)(sbits ^ 0x80000000u));
                    const uint32_t* gb = bp + (unsigned)g * 4u;
                    unsigned goff = (unsigned)g * 128u;
        #pragma clang loop unroll(disable)
            for (int c = 0; c < 4; c++) {
                        unsigned coff = goff + (unsigned)c * 32u;
                        HVX_Vector vWgt = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb[c]), vMask), vMask), vS, vNS);
                        vAcc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc0, Q6_Vqf32_vmpy_VsfVsf(vWgt, *(const HVX_Vector*)(x_b0 + coff)));
                        vAcc1 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc1, Q6_Vqf32_vmpy_VsfVsf(vWgt, *(const HVX_Vector*)(x_b1 + coff)));
                        vAcc2 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc2, Q6_Vqf32_vmpy_VsfVsf(vWgt, *(const HVX_Vector*)(x_b2 + coff)));
                        vAcc3 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc3, Q6_Vqf32_vmpy_VsfVsf(vWgt, *(const HVX_Vector*)(x_b3 + coff)));
                    }
                }
                y[(b0 + 0) * out_dim + i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc0));
                y[(b0 + 1) * out_dim + i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc1));
                y[(b0 + 2) * out_dim + i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc2));
                y[(b0 + 3) * out_dim + i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc3));
            }
        }
        for (; b0 < batch; b0++) {
            const float* x_b0 = hvx_xalign + (unsigned)b0 * (unsigned)in_dim;
            for (int i = r_base; i < r_end; i++) {
                const unsigned short* sg = scales + (unsigned)i * (unsigned)ng;
                const uint32_t* bp = bits_u32 + (unsigned)i * (unsigned)prow_u32;
                HVX_Vector vAcc = Q6_V_vzero();
                for (int g = 0; g < ng; g++) {
                    unsigned sbits = f16_scale_to_f32_bits_half(sg[g]);
                    HVX_Vector vS = Q6_V_vsplat_R((int)sbits), vNS = Q6_V_vsplat_R((int)(sbits ^ 0x80000000u));
                    const uint32_t* gb = bp + (unsigned)g * 4u;
                    unsigned goff = (unsigned)g * 128u;
        #pragma clang loop unroll(disable)
            for (int c = 0; c < 4; c++) {
                        unsigned coff = goff + (unsigned)c * 32u;
                        HVX_Vector vWgt = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb[c]), vMask), vMask), vS, vNS);
                        vAcc = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc, Q6_Vqf32_vmpy_VsfVsf(vWgt, *(const HVX_Vector*)(x_b0 + coff)));
                    }
                }
                y[b0 * out_dim + i] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc));
            }
        }
    }
}

// ============================================================================
// ============================================================================
// 4. Batched Ternary Q2 Slice (batch > 1) - Register-Blocked Dual-Token Kernel
//    Shares 100% of DRAM weight loads, bit-plane shifts/masks, and FP16 scale
//    unpacking across pairs of tokens (b0, b0+1) using 26 of 32 HVX registers.
// ============================================================================
static __attribute__((noinline)) void process_slice_q2_pair(int r0, int r1, int in_dim, int prow,
                                  const uint32_t* bits_u32, const unsigned short* scales,
                                  const signed char* qx8_0, const float* sx4_0, float* y_0,
                                  const signed char* qx8_1, const float* sx4_1, float* y_1) {
    int ng = in_dim >> 7;
    if (ng <= 0) return;
    int prow_u32 = prow >> 2;
    int half_prow_u32 = prow_u32 >> 1;
    uint32_t l2_bits_cfg = 0x00808000u | (uint32_t)((4 * prow) >> 7);
    HVX_Vector vOnes = Q6_V_vsplat_R(0x01010101);

    int i = r0;
    for (; i + 3 < r1; i += 4) {
        if (i + 7 < r1) {
            Q6_l2fetch_AR((void*)(bits_u32 + (unsigned)(i + 4) * (unsigned)prow_u32), (int)l2_bits_cfg);
        }
        const unsigned short* sg0 = scales + (unsigned)(i + 0) * (unsigned)ng;
        const unsigned short* sg1 = scales + (unsigned)(i + 1) * (unsigned)ng;
        const unsigned short* sg2 = scales + (unsigned)(i + 2) * (unsigned)ng;
        const unsigned short* sg3 = scales + (unsigned)(i + 3) * (unsigned)ng;

        const uint32_t* p_bp0 = bits_u32 + (unsigned)(i + 0) * (unsigned)prow_u32;
        const uint32_t* n_bp0 = p_bp0 + (unsigned)half_prow_u32;
        const uint32_t* p_bp1 = bits_u32 + (unsigned)(i + 1) * (unsigned)prow_u32;
        const uint32_t* n_bp1 = p_bp1 + (unsigned)half_prow_u32;
        const uint32_t* p_bp2 = bits_u32 + (unsigned)(i + 2) * (unsigned)prow_u32;
        const uint32_t* n_bp2 = p_bp2 + (unsigned)half_prow_u32;
        const uint32_t* p_bp3 = bits_u32 + (unsigned)(i + 3) * (unsigned)prow_u32;
        const uint32_t* n_bp3 = p_bp3 + (unsigned)half_prow_u32;

        int one_r = 0x01010101;
        asm volatile("" : "+r"(one_r));
        HVX_Vector vOneTmp = Q6_V_vsplat_R(one_r);
        HVX_Vector vAcc4_0 = Q6_Vqf32_vsub_VsfVsf(vOneTmp, vOneTmp);
        HVX_Vector vAcc4_1 = vAcc4_0;

        for (int g = 0; g < ng; g += 8) {
            unsigned goff_u32 = (unsigned)g * 4u;
            HVX_Vector vP0 = *(const HVX_Vector*)(p_bp0 + goff_u32);
            HVX_Vector vN0 = *(const HVX_Vector*)(n_bp0 + goff_u32);
            HVX_Vector vP1 = *(const HVX_Vector*)(p_bp1 + goff_u32);
            HVX_Vector vN1 = *(const HVX_Vector*)(n_bp1 + goff_u32);
            HVX_Vector vP2 = *(const HVX_Vector*)(p_bp2 + goff_u32);
            HVX_Vector vN2 = *(const HVX_Vector*)(n_bp2 + goff_u32);
            HVX_Vector vP3 = *(const HVX_Vector*)(p_bp3 + goff_u32);
            HVX_Vector vN3 = *(const HVX_Vector*)(n_bp3 + goff_u32);

            const HVX_Vector* vX_vec0 = (const HVX_Vector*)(qx8_0 + (unsigned)g * 128u);
            const HVX_Vector* vX_vec1 = (const HVX_Vector*)(qx8_1 + (unsigned)g * 128u);
            HVX_Vector vX0_a = vX_vec0[0];
            HVX_Vector vX0_b = vX_vec1[0];
            HVX_Vector vW0 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP0, vOnes), Q6_V_vand_VV(vN0, vOnes));
            HVX_Vector vW1 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP1, vOnes), Q6_V_vand_VV(vN1, vOnes));
            HVX_Vector vW2 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP2, vOnes), Q6_V_vand_VV(vN2, vOnes));
            HVX_Vector vW3 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP3, vOnes), Q6_V_vand_VV(vN3, vOnes));

            HVX_Vector vDot0_0 = Q6_Vw_vrmpy_VbVb(vW0, vX0_a);
            HVX_Vector vDot1_0 = Q6_Vw_vrmpy_VbVb(vW1, vX0_a);
            HVX_Vector vDot2_0 = Q6_Vw_vrmpy_VbVb(vW2, vX0_a);
            HVX_Vector vDot3_0 = Q6_Vw_vrmpy_VbVb(vW3, vX0_a);

            HVX_Vector vDot0_1 = Q6_Vw_vrmpy_VbVb(vW0, vX0_b);
            HVX_Vector vDot1_1 = Q6_Vw_vrmpy_VbVb(vW1, vX0_b);
            HVX_Vector vDot2_1 = Q6_Vw_vrmpy_VbVb(vW2, vX0_b);
            HVX_Vector vDot3_1 = Q6_Vw_vrmpy_VbVb(vW3, vX0_b);

#pragma clang loop unroll(full)
            for (int k = 1; k < 8; k++) {
                HVX_Vector vXk_a = vX_vec0[k];
                HVX_Vector vXk_b = vX_vec1[k];
                vP0 = Q6_Vuw_vlsr_VuwR(vP0, 1); vN0 = Q6_Vuw_vlsr_VuwR(vN0, 1);
                vP1 = Q6_Vuw_vlsr_VuwR(vP1, 1); vN1 = Q6_Vuw_vlsr_VuwR(vN1, 1);
                vP2 = Q6_Vuw_vlsr_VuwR(vP2, 1); vN2 = Q6_Vuw_vlsr_VuwR(vN2, 1);
                vP3 = Q6_Vuw_vlsr_VuwR(vP3, 1); vN3 = Q6_Vuw_vlsr_VuwR(vN3, 1);
                vW0 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP0, vOnes), Q6_V_vand_VV(vN0, vOnes));
                vW1 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP1, vOnes), Q6_V_vand_VV(vN1, vOnes));
                vW2 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP2, vOnes), Q6_V_vand_VV(vN2, vOnes));
                vW3 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vP3, vOnes), Q6_V_vand_VV(vN3, vOnes));

                vDot0_0 = Q6_Vw_vrmpyacc_VwVbVb(vDot0_0, vW0, vXk_a);
                vDot1_0 = Q6_Vw_vrmpyacc_VwVbVb(vDot1_0, vW1, vXk_a);
                vDot2_0 = Q6_Vw_vrmpyacc_VwVbVb(vDot2_0, vW2, vXk_a);
                vDot3_0 = Q6_Vw_vrmpyacc_VwVbVb(vDot3_0, vW3, vXk_a);

                vDot0_1 = Q6_Vw_vrmpyacc_VwVbVb(vDot0_1, vW0, vXk_b);
                vDot1_1 = Q6_Vw_vrmpyacc_VwVbVb(vDot1_1, vW1, vXk_b);
                vDot2_1 = Q6_Vw_vrmpyacc_VwVbVb(vDot2_1, vW2, vXk_b);
                vDot3_1 = Q6_Vw_vrmpyacc_VwVbVb(vDot3_1, vW3, vXk_b);
            }

            HVX_VectorPair W01_0   = Q6_W_vshuff_VVR(vDot1_0, vDot0_0, 4);
            HVX_Vector     v01_0   = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W01_0), Q6_V_hi_W(W01_0));
            HVX_VectorPair W23_0   = Q6_W_vshuff_VVR(vDot3_0, vDot2_0, 4);
            HVX_Vector     v23_0   = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W23_0), Q6_V_hi_W(W23_0));
            HVX_VectorPair W0123_0 = Q6_W_vshuff_VVR(v23_0, v01_0, 8);
            HVX_Vector     vDot4_0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W0123_0), Q6_V_hi_W(W0123_0));

            HVX_VectorPair W01_1   = Q6_W_vshuff_VVR(vDot1_1, vDot0_1, 4);
            HVX_Vector     v01_1   = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W01_1), Q6_V_hi_W(W01_1));
            HVX_VectorPair W23_1   = Q6_W_vshuff_VVR(vDot3_1, vDot2_1, 4);
            HVX_Vector     v23_1   = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W23_1), Q6_V_hi_W(W23_1));
            HVX_VectorPair W0123_1 = Q6_W_vshuff_VVR(v23_1, v01_1, 8);
            HVX_Vector     vDot4_1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(W0123_1), Q6_V_hi_W(W0123_1));

            asm volatile("" : "+v"(vDot4_0), "+v"(vDot4_1));

            uintptr_t a0 = (uintptr_t)(sg0 + g);
            uintptr_t a1 = (uintptr_t)(sg1 + g);
            uintptr_t a2 = (uintptr_t)(sg2 + g);
            uintptr_t a3 = (uintptr_t)(sg3 + g);

            HVX_Vector vS0 = Q6_V_vror_VR(*(const HVX_Vector*)(a0 & ~(uintptr_t)127), (int)(a0 & 127));
            HVX_Vector vS1 = Q6_V_vror_VR(*(const HVX_Vector*)(a1 & ~(uintptr_t)127), (int)(a1 & 127));
            HVX_Vector vS2 = Q6_V_vror_VR(*(const HVX_Vector*)(a2 & ~(uintptr_t)127), (int)(a2 & 127));
            HVX_Vector vS3 = Q6_V_vror_VR(*(const HVX_Vector*)(a3 & ~(uintptr_t)127), (int)(a3 & 127));

            HVX_Vector vS01   = Q6_V_lo_W(Q6_W_vshuff_VVR(vS1, vS0, -2));
            HVX_Vector vS23   = Q6_V_lo_W(Q6_W_vshuff_VVR(vS3, vS2, -2));
            HVX_Vector vS0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(vS23, vS01, -4));
            HVX_Vector vH     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(vS0123));

            int bias38_r = 0x38000000;
            asm volatile("" : "+r"(bias38_r));
            HVX_Vector vSW4   = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(vH, 13), Q6_V_vsplat_R(bias38_r));

            HVX_Vector vSX4_0   = *(const HVX_Vector*)(sx4_0 + (unsigned)g * 4u);
            HVX_Vector vDotSX_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(vDot4_0), vSX4_0));
            vAcc4_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc4_0, Q6_Vqf32_vmpy_VsfVsf(vDotSX_0, vSW4));

            HVX_Vector vSX4_1   = *(const HVX_Vector*)(sx4_1 + (unsigned)g * 4u);
            HVX_Vector vDotSX_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(vDot4_1), vSX4_1));
            vAcc4_1 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc4_1, Q6_Vqf32_vmpy_VsfVsf(vDotSX_1, vSW4));
        }
        {
            HVX_Vector vSum = Q6_Vsf_equals_Vqf32(vAcc4_0);
            vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 64));
            vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 32));
            vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 16));
            int b0 = hvx_vextract(vSum, 0);
            int b1 = hvx_vextract(vSum, 4);
            int b2 = hvx_vextract(vSum, 8);
            int b3 = hvx_vextract(vSum, 12);
            memcpy(&y_0[i + 0], &b0, 4);
            memcpy(&y_0[i + 1], &b1, 4);
            memcpy(&y_0[i + 2], &b2, 4);
            memcpy(&y_0[i + 3], &b3, 4);
        }
        {
            HVX_Vector vSum = Q6_Vsf_equals_Vqf32(vAcc4_1);
            vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 64));
            vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 32));
            vSum = Q6_Vsf_vadd_VsfVsf(vSum, Q6_V_vror_VR(vSum, 16));
            int b0 = hvx_vextract(vSum, 0);
            int b1 = hvx_vextract(vSum, 4);
            int b2 = hvx_vextract(vSum, 8);
            int b3 = hvx_vextract(vSum, 12);
            memcpy(&y_1[i + 0], &b0, 4);
            memcpy(&y_1[i + 1], &b1, 4);
            memcpy(&y_1[i + 2], &b2, 4);
            memcpy(&y_1[i + 3], &b3, 4);
        }
    }
    if (i < r1) {
        process_slice_q2_core(i, r1, in_dim, prow, bits_u32, scales, qx8_0, sx4_0, y_0);
        process_slice_q2_core(i, r1, in_dim, prow, bits_u32, scales, qx8_1, sx4_1, y_1);
    }
}

static void process_slice_batch_q2(int r0, int r1, int batch, int out_dim, int in_dim, int prow,
                                   const uint32_t* bits_u32, const unsigned short* scales,
                                   float* y) {
    int ng = in_dim >> 7;
    for (int r_base = r0; r_base < r1; r_base += 16) {
        int r_end = (r_base + 16 < r1) ? (r_base + 16) : r1;
        int b0 = 0;
        for (; b0 + 1 < batch; b0 += 2) {
            process_slice_q2_pair(r_base, r_end, in_dim, prow, bits_u32, scales,
                                  hvx_qx8 + (unsigned)(b0 + 0) * (unsigned)in_dim,
                                  hvx_sx4 + (unsigned)(b0 + 0) * (unsigned)(ng << 2),
                                  y + (unsigned)(b0 + 0) * (unsigned)out_dim,
                                  hvx_qx8 + (unsigned)(b0 + 1) * (unsigned)in_dim,
                                  hvx_sx4 + (unsigned)(b0 + 1) * (unsigned)(ng << 2),
                                  y + (unsigned)(b0 + 1) * (unsigned)out_dim);
        }
        for (; b0 < batch; b0++) {
            process_slice_q2_core(r_base, r_end, in_dim, prow, bits_u32, scales,
                                  hvx_qx8 + (unsigned)b0 * (unsigned)in_dim,
                                  hvx_sx4 + (unsigned)b0 * (unsigned)(ng << 2),
                                  y + (unsigned)b0 * (unsigned)out_dim);
        }
    }
}

static void dispatch_slice(int r0, int r1, int batch, int out_dim, int in_dim, int prow,
                           const uint32_t* bits_u32, const unsigned short* scales, float* y) {
    int is_ternary = (prow >= (in_dim >> 2));
    if (batch <= 1) {
        if (is_ternary) {
            process_slice_q2(r0, r1, in_dim, prow, bits_u32, scales, y);
        } else {
            process_slice_q1(r0, r1, in_dim, prow, bits_u32, scales, y);
        }
    } else {
        if (is_ternary) {
            process_slice_batch_q2(r0, r1, batch, out_dim, in_dim, prow, bits_u32, scales, y);
        } else {
            process_slice_batch_q1(r0, r1, batch, out_dim, in_dim, prow, bits_u32, scales, y);
        }
    }
}

static inline float fast_expf_dsp(float x) {
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

static inline float silu_f_dsp(float v) { return v / (1.0f + fast_expf_dsp(-v)); }

// 1024-bit HVX SIMD 32-float SiLU: v / (1.0f + exp(-v))
static inline HVX_Vector hvx_silu_vec(HVX_Vector vX) {
    HVX_Vector vNegX = Q6_Vsf_vfneg_Vsf(vX);
    vNegX = Q6_Vsf_vmax_VsfVsf(vNegX, Q6_V_vsplat_R((int)0xC2AE0000u)); // -87.0f
    vNegX = Q6_Vsf_vmin_VsfVsf(vNegX, Q6_V_vsplat_R((int)0x42A00000u)); // +80.0f

    HVX_Vector vZ      = Q6_Vsf_vmpy_VsfVsf(vNegX, Q6_V_vsplat_R((int)0x3FB8AA3Bu)); // log2(e)
    HVX_Vector vBias   = Q6_V_vsplat_R((int)0x4B400000u); // 12582912.0f
    HVX_Vector vBiased = Q6_Vsf_vadd_VsfVsf(vZ, vBias);
    HVX_Vector vKi     = Q6_Vw_vsub_VwVw(vBiased, vBias);
    HVX_Vector vKf     = Q6_Vsf_vsub_VsfVsf(vBiased, vBias);

    HVX_Vector vR = Q6_Vsf_vsub_VsfVsf(vNegX, Q6_Vsf_vmpy_VsfVsf(vKf, Q6_V_vsplat_R((int)0x3F317200u)));
    vR = Q6_Vsf_vsub_VsfVsf(vR, Q6_Vsf_vmpy_VsfVsf(vKf, Q6_V_vsplat_R((int)0x35BFBE8Eu)));

    HVX_Vector vOne = Q6_V_vsplat_R((int)0x3F800000u);
    HVX_Vector vP   = Q6_V_vsplat_R((int)0x39500D01u);
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3AB60B61u));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3C088889u));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3D2AAAAbu));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3E2AAAAbu));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3F000000u));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), vOne);
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), vOne);

    HVX_Vector vPow2 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(vKi, Q6_V_vsplat_R(127)), 23);
    HVX_Vector vExp  = Q6_Vsf_vmpy_VsfVsf(vP, vPow2);

    HVX_Vector vDen = Q6_Vsf_vadd_VsfVsf(vOne, vExp);
    HVX_Vector vTwo = Q6_V_vsplat_R((int)0x40000000u);
    HVX_Vector vY   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R((int)0x7EF127EAu), vDen);
    vY = Q6_Vsf_vmpy_VsfVsf(vY, Q6_Vsf_vsub_VsfVsf(vTwo, Q6_Vsf_vmpy_VsfVsf(vDen, vY)));
    vY = Q6_Vsf_vmpy_VsfVsf(vY, Q6_Vsf_vsub_VsfVsf(vTwo, Q6_Vsf_vmpy_VsfVsf(vDen, vY)));
    vY = Q6_Vsf_vmpy_VsfVsf(vY, Q6_Vsf_vsub_VsfVsf(vTwo, Q6_Vsf_vmpy_VsfVsf(vDen, vY)));
    return Q6_Vsf_vmpy_VsfVsf(vX, vY);
}

static const uint32_t hvx_lane_idx[32] __attribute__((aligned(128))) = {
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31
};

// 1024-bit HVX SIMD Fast Walsh-Hadamard Transform (1024 floats = 32 HVX vectors)
static void dsp_fwht1024_block(float* data, const float* signs) {
    HVX_UVector* vdata = (HVX_UVector*)data;
    const HVX_UVector* vsigns = (const HVX_UVector*)signs;
    HVX_Vector vIdx   = *(const HVX_Vector*)hvx_lane_idx;
    HVX_Vector vZero  = Q6_V_vzero();
    HVX_Vector vScale = Q6_V_vsplat_R((int)0x3D000000u); // 0.03125f = 1/32

    for (int half = 0; half < 2; half++) {
        int base = half * 16;
        HVX_Vector v[16];
        HVX_VectorPred q1 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vIdx, Q6_V_vsplat_R(1)), vZero);
        HVX_VectorPred q2 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vIdx, Q6_V_vsplat_R(2)), vZero);
        HVX_VectorPred q4 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vIdx, Q6_V_vsplat_R(4)), vZero);
        HVX_VectorPred q8 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vIdx, Q6_V_vsplat_R(8)), vZero);

        for (int m = 0; m < 16; m++) {
            HVX_Vector x = vdata[base + m];
            if (vsigns) {
                x = Q6_Vsf_vmpy_VsfVsf(x, vsigns[base + m]);
            }
            x = Q6_V_vmux_QVV(q1, Q6_Vsf_vadd_VsfVsf(x, Q6_V_vror_VR(x, 4)),
                                  Q6_Vsf_vsub_VsfVsf(Q6_V_vror_VR(x, 124), x));
            x = Q6_V_vmux_QVV(q2, Q6_Vsf_vadd_VsfVsf(x, Q6_V_vror_VR(x, 8)),
                                  Q6_Vsf_vsub_VsfVsf(Q6_V_vror_VR(x, 120), x));
            x = Q6_V_vmux_QVV(q4, Q6_Vsf_vadd_VsfVsf(x, Q6_V_vror_VR(x, 16)),
                                  Q6_Vsf_vsub_VsfVsf(Q6_V_vror_VR(x, 112), x));
            x = Q6_V_vmux_QVV(q8, Q6_Vsf_vadd_VsfVsf(x, Q6_V_vror_VR(x, 32)),
                                  Q6_Vsf_vsub_VsfVsf(Q6_V_vror_VR(x, 96), x));
            v[m] = x;
        }

        HVX_VectorPred q16 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vIdx, Q6_V_vsplat_R(16)), vZero);
        for (int m = 0; m < 16; m++) {
            HVX_Vector x = v[m];
            HVX_Vector xr = Q6_V_vror_VR(x, 64);
            v[m] = Q6_V_vmux_QVV(q16, Q6_Vsf_vadd_VsfVsf(x, xr), Q6_Vsf_vsub_VsfVsf(xr, x));
        }

        // Inter-vector butterfly stages s = 1, 2, 4, 8 (h = 32, 64, 128, 256)
        for (int step = 1; step < 16; step <<= 1) {
            for (int i = 0; i < 16; i += (step << 1)) {
                for (int k = 0; k < step; k++) {
                    HVX_Vector u = v[i + k];
                    HVX_Vector w = v[i + k + step];
                    v[i + k]        = Q6_Vsf_vadd_VsfVsf(u, w);
                    v[i + k + step] = Q6_Vsf_vsub_VsfVsf(u, w);
                }
            }
        }

        if (half == 0) {
            for (int m = 0; m < 16; m++) {
                vdata[m] = v[m];
            }
        } else {
            // Final stage s = 16 (h = 512) + scale by 1/32
            for (int m = 0; m < 16; m++) {
                HVX_Vector u = vdata[m];
                HVX_Vector w = v[m];
                vdata[m]      = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vadd_VsfVsf(u, w), vScale);
                vdata[m + 16] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(u, w), vScale);
            }
        }
    }
}

static inline float sigf_dsp(float x) { return 1.0f / (1.0f + fast_expf_dsp(-x)); }
static inline float log1pf_dsp(float x) {
    if (x <= -1.0f) return -87.0f;
    if (x > 100.0f) return logf(x);
    return log1pf(x);
}

static void run_lin_prep_worker(const WorkerTask* t) {
    const float* x     = t->x_in;
    const uint32_t* wa = t->wa;
    const uint32_t* wb = t->wb;
    float* avec = t->avec_out;
    float* bvec = t->bvec_out;
    const HVX_UVector* vx = (const HVX_UVector*)x;
    int mask_hi16_r = (int)0xFFFF0000u;
    asm volatile("" : "+r"(mask_hi16_r));
    HVX_Vector vMaskHi16 = Q6_V_vsplat_R(mask_hi16_r);

    for (int r = t->h0; r < t->h1; r++) {
        const HVX_UVector* vwa = (const HVX_UVector*)(wa + (size_t)r * 2560);
        const HVX_UVector* vwb = (const HVX_UVector*)(wb + (size_t)r * 2560);
        if (r + 1 < t->h1) {
            Q6_l2fetch_AR((void*)(wa + (size_t)(r + 1) * 2560), (int)0x00808050u);
            Q6_l2fetch_AR((void*)(wb + (size_t)(r + 1) * 2560), (int)0x00808050u);
        }
        HVX_Vector accA0 = Q6_V_vzero(), accA1 = Q6_V_vzero();
        HVX_Vector accB0 = Q6_V_vzero(), accB1 = Q6_V_vzero();
        for (int g = 0; g < 160; g += 4) {
            HVX_Vector x0 = vx[g+0], x1 = vx[g+1], x2 = vx[g+2], x3 = vx[g+3];
            HVX_Vector pA01 = vwa[(g >> 1) + 0];
            HVX_Vector pA23 = vwa[(g >> 1) + 1];
            HVX_Vector wa0 = Q6_Vw_vasl_VwR(pA01, 16);
            HVX_Vector wa1 = Q6_V_vand_VV(pA01, vMaskHi16);
            HVX_Vector wa2 = Q6_Vw_vasl_VwR(pA23, 16);
            HVX_Vector wa3 = Q6_V_vand_VV(pA23, vMaskHi16);
            HVX_Vector pa01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(wa0, x0), Q6_Vqf32_vmpy_VsfVsf(wa1, x1));
            HVX_Vector pa23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(wa2, x2), Q6_Vqf32_vmpy_VsfVsf(wa3, x3));
            accA0 = Q6_Vsf_vadd_VsfVsf(accA0, Q6_Vsf_equals_Vqf32(pa01));
            accA1 = Q6_Vsf_vadd_VsfVsf(accA1, Q6_Vsf_equals_Vqf32(pa23));

            HVX_Vector pB01 = vwb[(g >> 1) + 0];
            HVX_Vector pB23 = vwb[(g >> 1) + 1];
            HVX_Vector wb0 = Q6_Vw_vasl_VwR(pB01, 16);
            HVX_Vector wb1 = Q6_V_vand_VV(pB01, vMaskHi16);
            HVX_Vector wb2 = Q6_Vw_vasl_VwR(pB23, 16);
            HVX_Vector wb3 = Q6_V_vand_VV(pB23, vMaskHi16);
            HVX_Vector pb01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(wb0, x0), Q6_Vqf32_vmpy_VsfVsf(wb1, x1));
            HVX_Vector pb23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(wb2, x2), Q6_Vqf32_vmpy_VsfVsf(wb3, x3));
            accB0 = Q6_Vsf_vadd_VsfVsf(accB0, Q6_Vsf_equals_Vqf32(pb01));
            accB1 = Q6_Vsf_vadd_VsfVsf(accB1, Q6_Vsf_equals_Vqf32(pb23));
        }
        avec[r] = hvx_reduce_sum32(Q6_Vsf_vadd_VsfVsf(accA0, accA1));
        bvec[r] = hvx_reduce_sum32(Q6_Vsf_vadd_VsfVsf(accB0, accB1));
    }

    const float* cw = t->cw;
    float* conv = t->conv;
    float* qkv  = t->qkv;
    float acc_buf[32] __attribute__((aligned(128)));
    int c = t->c0;
    for (; c + 31 < t->c1; c += 32) {
        for (int k = 0; k < 32; k++) {
            int ch = c + k;
            float xm3 = conv[ch * 3 + 0];
            float xm2 = conv[ch * 3 + 1];
            float xm1 = conv[ch * 3 + 2];
            float x0  = qkv[ch];
            conv[ch * 3 + 0] = xm2;
            conv[ch * 3 + 1] = xm1;
            conv[ch * 3 + 2] = x0;
            const float* cwc = cw + (size_t)ch * 4;
            acc_buf[k] = xm3 * cwc[0] + xm2 * cwc[1] + xm1 * cwc[2] + x0 * cwc[3];
        }
        *(HVX_UVector*)(qkv + c) = hvx_silu_vec(*(const HVX_Vector*)acc_buf);
    }
    for (; c < t->c1; c++) {
        float xm3 = conv[c * 3 + 0];
        float xm2 = conv[c * 3 + 1];
        float xm1 = conv[c * 3 + 2];
        float x0  = qkv[c];
        conv[c * 3 + 0] = xm2;
        conv[c * 3 + 1] = xm1;
        conv[c * 3 + 2] = x0;
        const float* cwc = cw + (size_t)c * 4;
        float acc = xm3 * cwc[0] + xm2 * cwc[1] + xm1 * cwc[2] + x0 * cwc[3];
        qkv[c] = silu_f_dsp(acc);
    }
}

static void run_deltanet_worker(const WorkerTask* t) {
    int h0 = t->h0;
    int h1 = t->h1;
    const float* alog = t->alog;
    const float* dtb = t->dtb;
    const float* avec = t->avec;
    const float* bvec = t->bvec;
    const float* q = t->q;
    const float* k = t->k;
    const float* v = t->v;
    const float* z = t->z;
    const float* nw = t->nw;
    float* S = t->S;
    float* no = t->no;
    const HVX_UVector* vnw = (const HVX_UVector*)nw;

    for (int h = h0; h < h1; h++) {
        float oh[128] __attribute__((aligned(128)));
        float* Sh = S + (size_t)h * 128 * 128;
        float al = -fast_expf_dsp(alog[h]);
        float db = dtb[h];
        float beta = sigf_dsp(bvec[h]);
        float gv = al * log1pf_dsp(fast_expf_dsp(avec[h] + db));
        float eg = fast_expf_dsp(gv);
        const float* qh = q + (size_t)(h / 3) * 128;
        const float* kh = k + (size_t)(h / 3) * 128;
        const float* vh = v + (size_t)h * 128;

        int32_t eg_bits;
        memcpy(&eg_bits, &eg, 4);
        HVX_Vector vEg = Q6_V_vsplat_R(eg_bits);

        const HVX_UVector* vQh = (const HVX_UVector*)qh;
        const HVX_UVector* vKh = (const HVX_UVector*)kh;
        HVX_Vector vQ0 = vQh[0], vQ1 = vQh[1], vQ2 = vQh[2], vQ3 = vQh[3];
        HVX_Vector vK0 = vKh[0], vK1 = vKh[1], vK2 = vKh[2], vK3 = vKh[3];

        HVX_Vector vKQ01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vK0, vQ0), Q6_Vqf32_vmpy_VsfVsf(vK1, vQ1));
        HVX_Vector vKQ23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vK2, vQ2), Q6_Vqf32_vmpy_VsfVsf(vK3, vQ3));
        float kq = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vKQ01, vKQ23)));
        double norm_sq = 0.0;

        Q6_l2fetch_AR((void*)Sh, (int)0x00808010u);
        for (int j = 0; j < 128; j++) {
            if (j + 4 < 128) {
                Q6_l2fetch_AR((void*)(Sh + (size_t)(j + 4) * 128), (int)0x00808004u);
            }
            HVX_UVector* vSj = (HVX_UVector*)(Sh + (size_t)j * 128);
            HVX_Vector vS0 = vSj[0], vS1 = vSj[1], vS2 = vSj[2], vS3 = vSj[3];

            HVX_Vector vSK01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS0, vK0), Q6_Vqf32_vmpy_VsfVsf(vS1, vK1));
            HVX_Vector vSK23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS2, vK2), Q6_Vqf32_vmpy_VsfVsf(vS3, vK3));
            float sk = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vSK01, vSK23)));

            HVX_Vector vSQ01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS0, vQ0), Q6_Vqf32_vmpy_VsfVsf(vS1, vQ1));
            HVX_Vector vSQ23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS2, vQ2), Q6_Vqf32_vmpy_VsfVsf(vS3, vQ3));
            float sq = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vSQ01, vSQ23)));

            float kv = sk * eg;
            float delta = (vh[j] - kv) * beta;
            float oj = sq * eg + delta * kq;
            oh[j] = oj;
            norm_sq += (double)oj * oj;

            int32_t d_bits;
            memcpy(&d_bits, &delta, 4);
            HVX_Vector vDelta = Q6_V_vsplat_R(d_bits);

            vSj[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS0, vEg), Q6_Vqf32_vmpy_VsfVsf(vK0, vDelta)));
            vSj[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS1, vEg), Q6_Vqf32_vmpy_VsfVsf(vK1, vDelta)));
            vSj[2] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS2, vEg), Q6_Vqf32_vmpy_VsfVsf(vK2, vDelta)));
            vSj[3] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vS3, vEg), Q6_Vqf32_vmpy_VsfVsf(vK3, vDelta)));
        }
        const HVX_UVector* vzr = (const HVX_UVector*)(z + (size_t)h * 128);
        const HVX_Vector* voh  = (const HVX_Vector*)oh;
        HVX_UVector* vnoh      = (HVX_UVector*)(no + (size_t)h * 128);
        float inv = 1.0f / sqrtf((float)(norm_sq / 128.0) + 1e-6f);
        int32_t inv_bits;
        memcpy(&inv_bits, &inv, 4);
        HVX_Vector vInv = Q6_V_vsplat_R(inv_bits);
        for (int c = 0; c < 4; c++) {
            HVX_Vector vNormed = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(voh[c], vInv), vnw[c]);
            vnoh[c] = Q6_Vsf_vmpy_VsfVsf(vNormed, hvx_silu_vec(vzr[c]));
        }
    }
    // Fused FWHT-1024 on this worker's 8 heads (8 * 128 = 1024 floats) while hot in L1!
    dsp_fwht1024_block(no + (size_t)h0 * 128, t->fwht_signs);
}

static inline HVX_Vector hvx_mul_sigmoid_vec(HVX_Vector vO, HVX_Vector vX) {
    HVX_Vector vNegX = Q6_Vsf_vfneg_Vsf(vX);
    vNegX = Q6_Vsf_vmax_VsfVsf(vNegX, Q6_V_vsplat_R((int)0xC2AE0000u)); // -87.0f
    vNegX = Q6_Vsf_vmin_VsfVsf(vNegX, Q6_V_vsplat_R((int)0x42A00000u)); // +80.0f

    HVX_Vector vZ      = Q6_Vsf_vmpy_VsfVsf(vNegX, Q6_V_vsplat_R((int)0x3FB8AA3Bu)); // log2(e)
    HVX_Vector vBias   = Q6_V_vsplat_R((int)0x4B400000u); // 12582912.0f
    HVX_Vector vBiased = Q6_Vsf_vadd_VsfVsf(vZ, vBias);
    HVX_Vector vKi     = Q6_Vw_vsub_VwVw(vBiased, vBias);
    HVX_Vector vKf     = Q6_Vsf_vsub_VsfVsf(vBiased, vBias);

    HVX_Vector vR = Q6_Vsf_vsub_VsfVsf(vNegX, Q6_Vsf_vmpy_VsfVsf(vKf, Q6_V_vsplat_R((int)0x3F317200u)));
    vR = Q6_Vsf_vsub_VsfVsf(vR, Q6_Vsf_vmpy_VsfVsf(vKf, Q6_V_vsplat_R((int)0x35BFBE8Eu)));

    HVX_Vector vOne = Q6_V_vsplat_R((int)0x3F800000u);
    HVX_Vector vP   = Q6_V_vsplat_R((int)0x39500D01u);
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3AB60B61u));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3C088889u));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3D2AAAAbu));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3E2AAAAbu));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), Q6_V_vsplat_R((int)0x3F000000u));
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), vOne);
    vP = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vP, vR), vOne);

    HVX_Vector vPow2 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(vKi, Q6_V_vsplat_R(127)), 23);
    HVX_Vector vExp  = Q6_Vsf_vmpy_VsfVsf(vP, vPow2);

    HVX_Vector vDen = Q6_Vsf_vadd_VsfVsf(vOne, vExp);
    HVX_Vector vTwo = Q6_V_vsplat_R((int)0x40000000u);
    HVX_Vector vY   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R((int)0x7EF127EAu), vDen);
    vY = Q6_Vsf_vmpy_VsfVsf(vY, Q6_Vsf_vsub_VsfVsf(vTwo, Q6_Vsf_vmpy_VsfVsf(vDen, vY)));
    vY = Q6_Vsf_vmpy_VsfVsf(vY, Q6_Vsf_vsub_VsfVsf(vTwo, Q6_Vsf_vmpy_VsfVsf(vDen, vY)));
    vY = Q6_Vsf_vmpy_VsfVsf(vY, Q6_Vsf_vsub_VsfVsf(vTwo, Q6_Vsf_vmpy_VsfVsf(vDen, vY)));
    return Q6_Vsf_vmpy_VsfVsf(vO, vY);
}

static void dsp_q8_enc_bf16_256(const float* v, uint32_t* dst_pair, float* s) {
    float mx = 0.0f;
    for (int i = 0; i < 256; i++) {
        float a = v[i] < 0.0f ? -v[i] : v[i];
        if (a > mx) mx = a;
    }
    float sc = mx > 0.0f ? (mx / 127.0f) : 1.0f;
    *s = sc;
    float qf[256];
    for (int i = 0; i < 256; i++) {
        int t = (int)(v[i] / sc + (v[i] >= 0.0f ? 0.5f : -0.5f));
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        qf[i] = (float)t;
    }
    const uint32_t* qu = (const uint32_t*)qf;
    for (int g = 0; g < 8; g += 2) {
        const uint32_t* s0 = qu + (g + 0) * 32;
        const uint32_t* s1 = qu + (g + 1) * 32;
        uint32_t* d = dst_pair + (g >> 1) * 32;
        for (int k = 0; k < 32; k++) {
            d[k] = (s1[k] & 0xFFFF0000u) | (s0[k] >> 16);
        }
    }
}

static void dsp_q4_enc_128b(const float* v, uint32_t* dst_packed, float* s) {
    float mx = 0.0f;
    for (int i = 0; i < 256; i++) {
        float a = v[i] < 0.0f ? -v[i] : v[i];
        if (a > mx) mx = a;
    }
    float sc = (mx > 0.0f) ? (mx / 7.0f) : 1.0f;
    *s = sc;
    float inv_sc = 1.0f / sc;
    int q[256];
    for (int i = 0; i < 256; i++) {
        int t = (int)(v[i] * inv_sc + (v[i] >= 0.0f ? 0.5f : -0.5f));
        if (t > 7) t = 7;
        if (t < -7) t = -7;
        q[i] = t;
    }
    for (int k = 0; k < 32; k++) {
        uint32_t w = 0;
        for (int g = 0; g < 8; g++) {
            uint32_t nib = (uint32_t)(q[g * 32 + k] & 0x0F);
            w |= (nib << (g * 4));
        }
        dst_packed[k] = w;
    }
}

static void run_full_gqa_worker(const WorkerTask* t) {
    int h0 = t->h0;
    int h1 = t->h1;
    int seqlen = t->c0;
    const float* qkv = t->qkv;
    const HVX_UVector* vQnw = (const HVX_UVector*)t->nw;
    const HVX_UVector* vRope = (const HVX_UVector*)t->cw;
    HVX_Vector vCos = vRope[0];
    HVX_Vector vSin = vRope[1];
    int mask_hi16_r = (int)0xFFFF0000u;
    asm volatile("" : "+r"(mask_hi16_r));
    HVX_Vector vMaskHi16 = Q6_V_vsplat_R(mask_hi16_r);

    for (int hq = h0; hq < h1; hq++) {
        const float* qq_h   = qkv + (size_t)hq * 512;
        const float* gate_h = qq_h + 256;
        const HVX_UVector* vQ_in = (const HVX_UVector*)qq_h;
        HVX_Vector vQ0 = vQ_in[0], vQ1 = vQ_in[1], vQ2 = vQ_in[2], vQ3 = vQ_in[3];
        HVX_Vector vQ4 = vQ_in[4], vQ5 = vQ_in[5], vQ6 = vQ_in[6], vQ7 = vQ_in[7];

        HVX_Vector s01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ0, vQ0), Q6_Vqf32_vmpy_VsfVsf(vQ1, vQ1));
        HVX_Vector s23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ2, vQ2), Q6_Vqf32_vmpy_VsfVsf(vQ3, vQ3));
        HVX_Vector s45 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ4, vQ4), Q6_Vqf32_vmpy_VsfVsf(vQ5, vQ5));
        HVX_Vector s67 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ6, vQ6), Q6_Vqf32_vmpy_VsfVsf(vQ7, vQ7));
        HVX_Vector s03 = Q6_Vqf32_vadd_Vqf32Vqf32(s01, s23);
        HVX_Vector s47 = Q6_Vqf32_vadd_Vqf32Vqf32(s45, s67);
        float sq = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(s03, s47)));
        float inv = 1.0f / sqrtf(sq * (1.0f / 256.0f) + 1e-6f);
        int32_t inv_bits;
        memcpy(&inv_bits, &inv, 4);
        HVX_Vector vInv = Q6_V_vsplat_R(inv_bits);

        vQ0 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ0, vInv), vQnw[0]);
        vQ1 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ1, vInv), vQnw[1]);
        vQ2 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ2, vInv), vQnw[2]);
        vQ3 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ3, vInv), vQnw[3]);
        vQ4 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ4, vInv), vQnw[4]);
        vQ5 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ5, vInv), vQnw[5]);
        vQ6 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ6, vInv), vQnw[6]);
        vQ7 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vQ7, vInv), vQnw[7]);

        // NeoX RoPE on first 64 dims (vQ0 = 0..31, vQ1 = 32..63)
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ0, vCos), Q6_Vqf32_vmpy_VsfVsf(vQ1, vSin)));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ0, vSin), Q6_Vqf32_vmpy_VsfVsf(vQ1, vCos)));
        vQ0 = r0;
        vQ1 = r1;

        int hk = hq / 6;
        size_t kv_stride = g_dsp_turbo4 ? 32 : 128;
        const uint32_t* K_hk = t->wa + (size_t)hk * g_dsp_kv_ctx_max * kv_stride;
        const uint32_t* V_hk = t->wb + (size_t)hk * g_dsp_kv_ctx_max * kv_stride;
        const float* KS_hk   = t->alog + (size_t)hk * g_dsp_kv_ctx_max;
        const float* VS_hk   = t->dtb  + (size_t)hk * g_dsp_kv_ctx_max;

        float* sc = t->sc;
        float mx = -1e30f;
        if (g_dsp_turbo4) {
            for (int tok = 0; tok < seqlen; tok++) {
                HVX_Vector vKt = *(const HVX_Vector*)(K_hk + (size_t)tok * 32);

                HVX_Vector k0 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt, 28), 28));
                HVX_Vector k1 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt, 24), 28));
                HVX_Vector k2 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt, 20), 28));
                HVX_Vector k3 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt, 16), 28));
                HVX_Vector k4 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt, 12), 28));
                HVX_Vector k5 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt,  8), 28));
                HVX_Vector k6 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vKt,  4), 28));
                HVX_Vector k7 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(vKt, 28));

                HVX_Vector d01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ0, k0), Q6_Vqf32_vmpy_VsfVsf(vQ1, k1));
                HVX_Vector d23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ2, k2), Q6_Vqf32_vmpy_VsfVsf(vQ3, k3));
                HVX_Vector d45 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ4, k4), Q6_Vqf32_vmpy_VsfVsf(vQ5, k5));
                HVX_Vector d67 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ6, k6), Q6_Vqf32_vmpy_VsfVsf(vQ7, k7));
                HVX_Vector d03 = Q6_Vqf32_vadd_Vqf32Vqf32(d01, d23);
                HVX_Vector d47 = Q6_Vqf32_vadd_Vqf32Vqf32(d45, d67);
                float dot = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(d03, d47)));
                float score = dot * (KS_hk[tok] * 0.0625f);
                sc[tok] = score;
                if (score > mx) mx = score;
            }
        } else {
            for (int tok = 0; tok < seqlen; tok++) {
                const HVX_UVector* vKt = (const HVX_UVector*)(K_hk + (size_t)tok * 128);
                HVX_Vector p01 = vKt[0], p23 = vKt[1], p45 = vKt[2], p67 = vKt[3];
                HVX_Vector k0 = Q6_Vw_vasl_VwR(p01, 16), k1 = Q6_V_vand_VV(p01, vMaskHi16);
                HVX_Vector k2 = Q6_Vw_vasl_VwR(p23, 16), k3 = Q6_V_vand_VV(p23, vMaskHi16);
                HVX_Vector k4 = Q6_Vw_vasl_VwR(p45, 16), k5 = Q6_V_vand_VV(p45, vMaskHi16);
                HVX_Vector k6 = Q6_Vw_vasl_VwR(p67, 16), k7 = Q6_V_vand_VV(p67, vMaskHi16);

                HVX_Vector d01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ0, k0), Q6_Vqf32_vmpy_VsfVsf(vQ1, k1));
                HVX_Vector d23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ2, k2), Q6_Vqf32_vmpy_VsfVsf(vQ3, k3));
                HVX_Vector d45 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ4, k4), Q6_Vqf32_vmpy_VsfVsf(vQ5, k5));
                HVX_Vector d67 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ6, k6), Q6_Vqf32_vmpy_VsfVsf(vQ7, k7));
                HVX_Vector d03 = Q6_Vqf32_vadd_Vqf32Vqf32(d01, d23);
                HVX_Vector d47 = Q6_Vqf32_vadd_Vqf32Vqf32(d45, d67);
                float dot = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(d03, d47)));
                float score = dot * (KS_hk[tok] * 0.0625f);
                sc[tok] = score;
                if (score > mx) mx = score;
            }
        }

        double sum_exp = 0.0;
        for (int tok = 0; tok < seqlen; tok++) {
            float e = fast_expf_dsp(sc[tok] - mx);
            sc[tok] = e;
            sum_exp += (double)e;
        }
        float inv_sum = (float)(1.0 / sum_exp);

        HVX_Vector vO0 = Q6_V_vzero(), vO1 = Q6_V_vzero(), vO2 = Q6_V_vzero(), vO3 = Q6_V_vzero();
        HVX_Vector vO4 = Q6_V_vzero(), vO5 = Q6_V_vzero(), vO6 = Q6_V_vzero(), vO7 = Q6_V_vzero();
        if (g_dsp_turbo4) {
            for (int tok = 0; tok < seqlen; tok++) {
                float p = (sc[tok] * inv_sum) * VS_hk[tok];
                int32_t p_bits;
                memcpy(&p_bits, &p, 4);
                HVX_Vector vP = Q6_V_vsplat_R(p_bits);
                HVX_Vector vVt = *(const HVX_Vector*)(V_hk + (size_t)tok * 32);

                HVX_Vector v0 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt, 28), 28));
                HVX_Vector v1 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt, 24), 28));
                HVX_Vector v2 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt, 20), 28));
                HVX_Vector v3 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt, 16), 28));
                HVX_Vector v4 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt, 12), 28));
                HVX_Vector v5 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt,  8), 28));
                HVX_Vector v6 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(vVt,  4), 28));
                HVX_Vector v7 = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(vVt, 28));

                vO0 = Q6_Vsf_vadd_VsfVsf(vO0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, vP)));
                vO1 = Q6_Vsf_vadd_VsfVsf(vO1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, vP)));
                vO2 = Q6_Vsf_vadd_VsfVsf(vO2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v2, vP)));
                vO3 = Q6_Vsf_vadd_VsfVsf(vO3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, vP)));
                vO4 = Q6_Vsf_vadd_VsfVsf(vO4, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v4, vP)));
                vO5 = Q6_Vsf_vadd_VsfVsf(vO5, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v5, vP)));
                vO6 = Q6_Vsf_vadd_VsfVsf(vO6, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v6, vP)));
                vO7 = Q6_Vsf_vadd_VsfVsf(vO7, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v7, vP)));
            }
        } else {
            for (int tok = 0; tok < seqlen; tok++) {
                float p = (sc[tok] * inv_sum) * VS_hk[tok];
                int32_t p_bits;
                memcpy(&p_bits, &p, 4);
                HVX_Vector vP = Q6_V_vsplat_R(p_bits);
                const HVX_UVector* vVt = (const HVX_UVector*)(V_hk + (size_t)tok * 128);
                HVX_Vector p01 = vVt[0], p23 = vVt[1], p45 = vVt[2], p67 = vVt[3];
                vO0 = Q6_Vsf_vadd_VsfVsf(vO0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vw_vasl_VwR(p01, 16), vP)));
                vO1 = Q6_Vsf_vadd_VsfVsf(vO1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_vand_VV(p01, vMaskHi16), vP)));
                vO2 = Q6_Vsf_vadd_VsfVsf(vO2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vw_vasl_VwR(p23, 16), vP)));
                vO3 = Q6_Vsf_vadd_VsfVsf(vO3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_vand_VV(p23, vMaskHi16), vP)));
                vO4 = Q6_Vsf_vadd_VsfVsf(vO4, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vw_vasl_VwR(p45, 16), vP)));
                vO5 = Q6_Vsf_vadd_VsfVsf(vO5, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_vand_VV(p45, vMaskHi16), vP)));
                vO6 = Q6_Vsf_vadd_VsfVsf(vO6, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vw_vasl_VwR(p67, 16), vP)));
                vO7 = Q6_Vsf_vadd_VsfVsf(vO7, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_vand_VV(p67, vMaskHi16), vP)));
            }
        }


        const HVX_UVector* vGate = (const HVX_UVector*)gate_h;
        HVX_UVector* vOut = (HVX_UVector*)(t->no + (size_t)hq * 256);
        vOut[0] = hvx_mul_sigmoid_vec(vO0, vGate[0]);
        vOut[1] = hvx_mul_sigmoid_vec(vO1, vGate[1]);
        vOut[2] = hvx_mul_sigmoid_vec(vO2, vGate[2]);
        vOut[3] = hvx_mul_sigmoid_vec(vO3, vGate[3]);
        vOut[4] = hvx_mul_sigmoid_vec(vO4, vGate[4]);
        vOut[5] = hvx_mul_sigmoid_vec(vO5, vGate[5]);
        vOut[6] = hvx_mul_sigmoid_vec(vO6, vGate[6]);
        vOut[7] = hvx_mul_sigmoid_vec(vO7, vGate[7]);
    }

    // Fused FWHT-1024 on this worker's 4 Q heads (4 * 256 = 1024 floats) while hot in L1!
    if (t->fwht_signs) {
        dsp_fwht1024_block(t->no + (size_t)h0 * 256, t->fwht_signs);
    }
}

static void worker_entry(void* arg) {
    int id = (int)(uintptr_t)arg;
    while (!g_pool_exit) {
        qurt_sem_down(&g_sem_start[id]);
        if (g_pool_exit) break;
        WorkerTask* t = &g_tasks[id];
        if (t->task_type == TASK_SWIGLU_FWHT) {
            for (int blk = t->blk_start; blk < t->blk_end; blk++) {
                int tok_idx = blk / 17;
                int blk_in_tok = blk % 17;
                int off_gate = tok_idx * 34816 + blk_in_tok * 1024;
                int off_up   = tok_idx * 34816 + 17408 + blk_in_tok * 1024;
                int off_out  = tok_idx * 17408 + blk_in_tok * 1024;
                int off_sign = blk_in_tok * 1024;

                const HVX_UVector* vgb = (const HVX_UVector*)(t->gate_up + off_gate);
                const HVX_UVector* vub = (const HVX_UVector*)(t->gate_up + off_up);
                float* ob              = t->out + off_out;
                HVX_UVector* vob       = (HVX_UVector*)ob;
                const float* sb        = t->signs ? (t->signs + off_sign) : NULL;

                for (int g = 0; g < 32; g++) {
                    vob[g] = Q6_Vsf_vmpy_VsfVsf(hvx_silu_vec(vgb[g]), vub[g]);
                }
                dsp_fwht1024_block(ob, sb);
            }
        } else if (t->task_type == TASK_DELTANET) {
            run_deltanet_worker(t);
        } else if (t->task_type == TASK_LIN_PREP) {
            run_lin_prep_worker(t);
        } else if (t->task_type == TASK_FULL_GQA) {
            run_full_gqa_worker(t);
        } else if (t->task_type == TASK_FWHT_BLOCK) {
            dsp_fwht1024_block(t->fwht_data, t->fwht_signs);
        } else {
            if (t->r0 < t->r1) {
                dispatch_slice(t->r0, t->r1, t->batch, t->out_dim, t->in_dim, t->prow, t->bits_u32, t->scales, t->y);
            }
        }
        qurt_sem_up(&g_sem_done[id]);
    }
}

static void init_thread_pool(void) {
    if (g_pool_ready) return;
    g_pool_exit = 0;
    qurt_thread_attr_t attr;
    for (int i = 0; i < NUM_WORKERS; i++) {
        qurt_sem_init_val(&g_sem_start[i], 0);
        qurt_sem_init_val(&g_sem_done[i], 0);
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
        qurt_thread_attr_set_stack_addr(&attr, g_stack[i]);
        qurt_thread_attr_set_priority(&attr, 50);
        qurt_thread_create(&g_tids[i], &attr, worker_entry, (void*)(uintptr_t)i);
    }
    g_pool_ready = 1;
}

int hvx_gemv_q1(int out_dim, int in_dim, int prow,
                const float* x, const unsigned char* bits,
                const short* scales, float* y) {
    if (!x || !bits || !scales || !y) return -1;
    if (out_dim <= 0 || in_dim <= 0 || (in_dim & 127) != 0 || prow <= 0) return -2;
    if (in_dim > HVX_MAX_K) return -3;

    init_thread_pool();

    if (prow >= (in_dim >> 2)) {
        quantize_x_q8_permuted(x, 1, in_dim);
    } else {
        memcpy(hvx_xalign, x, (unsigned)in_dim * 4u);
    }
    const uint32_t* bits_u32 = (const uint32_t*)bits;
    const unsigned short* sc = (const unsigned short*)scales;

    if (out_dim < 16) {
        dispatch_slice(0, out_dim, 1, out_dim, in_dim, prow, bits_u32, sc, y);
        return 0;
    }

    int chunk = (((out_dim + NUM_WORKERS - 1) / NUM_WORKERS) + 3) & ~3;
    for (int w = 0; w < NUM_WORKERS; w++) {
        int r0 = w * chunk;
        int r1 = (w + 1) * chunk;
        if (r0 > out_dim) r0 = out_dim;
        if (r1 > out_dim) r1 = out_dim;
        g_tasks[w].task_type = TASK_GEMV;
        g_tasks[w].r0 = r0;
        g_tasks[w].r1 = r1;
        g_tasks[w].batch = 1;
        g_tasks[w].out_dim = out_dim;
        g_tasks[w].in_dim = in_dim;
        g_tasks[w].prow = prow;
        g_tasks[w].bits_u32 = bits_u32;
        g_tasks[w].scales = sc;
        g_tasks[w].y = y;
        qurt_sem_up(&g_sem_start[w]);
    }

    for (int w = 0; w < NUM_WORKERS; w++) {
        qurt_sem_down(&g_sem_done[w]);
    }

    return 0;
}

int hvx_gemm_q1(int batch, int out_dim, int in_dim, int prow,
                const float* x, const unsigned char* bits,
                const short* scales, float* y) {
    if (batch <= 1) return hvx_gemv_q1(out_dim, in_dim, prow, x, bits, scales, y);
    if (!x || !bits || !scales || !y) return -1;
    if (out_dim <= 0 || in_dim <= 0 || (in_dim & 127) != 0 || prow <= 0) return -2;
    if ((size_t)batch * (size_t)in_dim > sizeof(hvx_xalign) / 4) return -3;

    init_thread_pool();

    if (prow >= (in_dim >> 2)) {
        quantize_x_q8_permuted(x, batch, in_dim);
    } else {
        memcpy(hvx_xalign, x, (unsigned)batch * (unsigned)in_dim * 4u);
    }
    const uint32_t* bits_u32 = (const uint32_t*)bits;
    const unsigned short* sc = (const unsigned short*)scales;

    int chunk = (((out_dim + NUM_WORKERS - 1) / NUM_WORKERS) + 3) & ~3;
    for (int w = 0; w < NUM_WORKERS; w++) {
        int r0 = w * chunk;
        int r1 = (w + 1) * chunk;
        if (r0 > out_dim) r0 = out_dim;
        if (r1 > out_dim) r1 = out_dim;
        g_tasks[w].task_type = TASK_GEMV;
        g_tasks[w].r0 = r0;
        g_tasks[w].r1 = r1;
        g_tasks[w].batch = batch;
        g_tasks[w].out_dim = out_dim;
        g_tasks[w].in_dim = in_dim;
        g_tasks[w].prow = prow;
        g_tasks[w].bits_u32 = bits_u32;
        g_tasks[w].scales = sc;
        g_tasks[w].y = y;
        qurt_sem_up(&g_sem_start[w]);
    }

    for (int w = 0; w < NUM_WORKERS; w++) {
        qurt_sem_down(&g_sem_done[w]);
    }

    return 0;
}

int hvx_swiglu_fwht1024(int batch, const float* gate_up, const float* signs, float* out) {
    init_thread_pool();
    int total_blocks = batch * 17;
    int blocks_per_worker = (total_blocks + NUM_WORKERS - 1) / NUM_WORKERS;

    for (int w = 0; w < NUM_WORKERS; w++) {
        int b0 = w * blocks_per_worker;
        int b1 = b0 + blocks_per_worker;
        if (b0 > total_blocks) b0 = total_blocks;
        if (b1 > total_blocks) b1 = total_blocks;

        g_tasks[w].task_type = TASK_SWIGLU_FWHT;
        g_tasks[w].blk_start = b0;
        g_tasks[w].blk_end = b1;
        g_tasks[w].gate_up = gate_up;
        g_tasks[w].signs = signs;
        g_tasks[w].out = out;
        qurt_sem_up(&g_sem_start[w]);
    }

    for (int w = 0; w < NUM_WORKERS; w++) {
        qurt_sem_down(&g_sem_done[w]);
    }

    return 0;
}

// ============================================================================
// Fused Linear Attention Block:
//   Input FWHT-5120 -> in_proj GEMV (16384x5120) -> in_proj_a/b dot products ->
//   Depthwise Conv1D (10240x4 taps) + SiLU -> Q/K RMSNorm ->
//   DeltaNet Recurrence S_h (48 heads x 128x128) + Fused FWHT-6144 across 6 QuRT threads ->
//   out_proj GEMV (5120x6144)
// ============================================================================
static float g_dsp_lin_x_fwht[5120] __attribute__((aligned(128)));
static float g_dsp_lin_qkv_z[16384] __attribute__((aligned(128)));
static float g_dsp_lin_no[6144] __attribute__((aligned(128)));

int hvx_lmhead_static_q1(int out_dim, int in_dim, int prow,
                         const float* x, const unsigned char* bits,
                         const short* scales, const float* signs_5120, float* y) {
    memcpy(g_dsp_lin_x_fwht, x, 5120 * sizeof(float));
    if (signs_5120) {
        for (int w = 0; w < 5; w++) {
            dsp_fwht1024_block(g_dsp_lin_x_fwht + w * 1024, signs_5120 + w * 1024);
        }
    }
    return hvx_gemv_q1(out_dim, in_dim, prow, g_dsp_lin_x_fwht, bits, scales, y);
}

static void dsp_rmsnorm_5120(const float* x, const float* w, float* out) {
    const HVX_UVector* vx = (const HVX_UVector*)x;
    const HVX_UVector* vw = (const HVX_UVector*)w;
    HVX_UVector* vout = (HVX_UVector*)out;
    HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero();
    for (int g = 0; g < 160; g += 4) {
        HVX_Vector x0 = vx[g+0], x1 = vx[g+1], x2 = vx[g+2], x3 = vx[g+3];
        HVX_Vector p01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, x0), Q6_Vqf32_vmpy_VsfVsf(x1, x1));
        HVX_Vector p23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, x2), Q6_Vqf32_vmpy_VsfVsf(x3, x3));
        acc0 = Q6_Vsf_vadd_VsfVsf(acc0, Q6_Vsf_equals_Vqf32(p01));
        acc1 = Q6_Vsf_vadd_VsfVsf(acc1, Q6_Vsf_equals_Vqf32(p23));
    }
    float s = hvx_reduce_sum32(Q6_Vsf_vadd_VsfVsf(acc0, acc1));
    float inv = 1.0f / sqrtf(s * (1.0f / 5120.0f) + 1e-6f);
    int32_t inv_bits;
    memcpy(&inv_bits, &inv, 4);
    HVX_Vector vInv = Q6_V_vsplat_R(inv_bits);
    for (int g = 0; g < 160; g++) {
        vout[g] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vx[g], vInv), vw[g]);
    }
}

int hvx_lin_attn_fused(int layer_idx,
                       const float* x,
                       const unsigned char* in_bits, const short* in_scales,
                       const unsigned char* out_bits, const short* out_scales,
                       const float* ssm_conv, const float* ssm_rec, const uint8_t* lin_aux,
                       const float* signs_5120, const float* signs_6144,
                       float* y) {
    init_thread_pool();

    if (layer_idx < 0) {
        int fi = -1 - layer_idx;
        const float* in_norm = (const float*)(lin_aux + (size_t)48 * 1188736 + (size_t)fi * 43008);
        dsp_rmsnorm_5120(x, in_norm, g_dsp_lin_x_fwht);
        if (signs_5120) {
            for (int w = 0; w < 5; w++) {
                dsp_fwht1024_block(g_dsp_lin_x_fwht + w * 1024, signs_5120 + w * 1024);
            }
        }
        return hvx_gemv_q1(14336, 5120, 1280, g_dsp_lin_x_fwht, in_bits, in_scales, y);
    }

    // Step 1: HVX Input FWHT-5120 directly in L1 cache (zero semaphore barrier!)
    memcpy(g_dsp_lin_x_fwht, x, 5120 * sizeof(float));
    if (signs_5120) {
        for (int w = 0; w < 5; w++) {
            dsp_fwht1024_block(g_dsp_lin_x_fwht + w * 1024, signs_5120 + w * 1024);
        }
    }

    // Step 2: in_proj GEMV (16384 x 5120, prow = 1280)
    int rc = hvx_gemv_q1(16384, 5120, 1280, g_dsp_lin_x_fwht, in_bits, in_scales, g_dsp_lin_qkv_z);
    if (rc != 0) return rc;

    float* qkv = g_dsp_lin_qkv_z;
    float* z   = g_dsp_lin_qkv_z + 10240;

    // Step 3 & 4: Parallel HVX in_proj_a/b dot products [48 x 5120] + Conv1D + SiLU [10240] across 6 QuRT threads
    const uint8_t* aux_layer = lin_aux + (size_t)layer_idx * 1188736;
    const uint32_t* wa = (const uint32_t*)(aux_layer + 0);
    const uint32_t* wb = (const uint32_t*)(aux_layer + 491520);
    const float* cw    = (const float*)(aux_layer + 983040);
    float* conv = (float*)ssm_conv + (size_t)layer_idx * 10240 * 3;
    float avec[48] __attribute__((aligned(128)));
    float bvec[48] __attribute__((aligned(128)));

    // 10240 = 320 * 32 -> 54 * 32 = 1728 channels per worker (aligned to 32-float HVX vectors!)
    int c_chunk = 54 * 32;
    for (int w = 0; w < NUM_WORKERS; w++) {
        int c0 = w * c_chunk;
        int c1 = (w + 1) * c_chunk;
        if (c0 > 10240) c0 = 10240;
        if (c1 > 10240) c1 = 10240;
        g_tasks[w].task_type = TASK_LIN_PREP;
        g_tasks[w].h0 = w * 8;
        g_tasks[w].h1 = (w + 1) * 8;
        g_tasks[w].x_in = x;
        g_tasks[w].wa = wa;
        g_tasks[w].wb = wb;
        g_tasks[w].avec_out = avec;
        g_tasks[w].bvec_out = bvec;
        g_tasks[w].cw = cw;
        g_tasks[w].conv = conv;
        g_tasks[w].qkv = qkv;
        g_tasks[w].c0 = c0;
        g_tasks[w].c1 = c1;
        qurt_sem_up(&g_sem_start[w]);
    }
    for (int w = 0; w < NUM_WORKERS; w++) {
        qurt_sem_down(&g_sem_done[w]);
    }

    float* q = qkv;
    float* k = qkv + 2048;
    float* v = qkv + 4096;

    // Step 5: Q / K RMSNorm on 16 heads (HVX 4-vector norm)
    for (int h = 0; h < 16; h++) {
        const HVX_UVector* vQ = (const HVX_UVector*)(q + h * 128);
        const HVX_UVector* vK = (const HVX_UVector*)(k + h * 128);
        HVX_Vector vQ01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ[0], vQ[0]), Q6_Vqf32_vmpy_VsfVsf(vQ[1], vQ[1]));
        HVX_Vector vQ23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vQ[2], vQ[2]), Q6_Vqf32_vmpy_VsfVsf(vQ[3], vQ[3]));
        float sq = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vQ01, vQ23)));

        HVX_Vector vK01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vK[0], vK[0]), Q6_Vqf32_vmpy_VsfVsf(vK[1], vK[1]));
        HVX_Vector vK23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vK[2], vK[2]), Q6_Vqf32_vmpy_VsfVsf(vK[3], vK[3]));
        float sk = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vK01, vK23)));

        float iq = (1.0f / sqrtf(sq + 1e-6f)) * 0.08838834765f;
        float ik = 1.0f / sqrtf(sk + 1e-6f);
        int32_t iq_bits, ik_bits;
        memcpy(&iq_bits, &iq, 4);
        memcpy(&ik_bits, &ik, 4);
        HVX_Vector vIQ = Q6_V_vsplat_R(iq_bits);
        HVX_Vector vIK = Q6_V_vsplat_R(ik_bits);
        HVX_UVector* vQ_out = (HVX_UVector*)(q + h * 128);
        HVX_UVector* vK_out = (HVX_UVector*)(k + h * 128);
        for (int c = 0; c < 4; c++) {
            vQ_out[c] = Q6_Vsf_vmpy_VsfVsf(vQ[c], vIQ);
            vK_out[c] = Q6_Vsf_vmpy_VsfVsf(vK[c], vIK);
        }
    }

    // Step 6 & 7 Fused: DeltaNet Recurrence + Fused FWHT-6144 across 6 QuRT threads
    const float* alog = (const float*)(aux_layer + 1146880);
    const float* dtb  = (const float*)(aux_layer + 1147072);
    const float* nw   = (const float*)(aux_layer + 1147264);
    float* S_layer    = (float*)ssm_rec + (size_t)layer_idx * 48 * 128 * 128;
    float* no         = g_dsp_lin_no;

    for (int w = 0; w < NUM_WORKERS; w++) {
        g_tasks[w].task_type = TASK_DELTANET;
        g_tasks[w].h0 = w * 8;
        g_tasks[w].h1 = (w + 1) * 8;
        g_tasks[w].alog = alog;
        g_tasks[w].dtb = dtb;
        g_tasks[w].avec = avec;
        g_tasks[w].bvec = bvec;
        g_tasks[w].q = q;
        g_tasks[w].k = k;
        g_tasks[w].v = v;
        g_tasks[w].z = z;
        g_tasks[w].nw = nw;
        g_tasks[w].S = S_layer;
        g_tasks[w].no = no;
        g_tasks[w].fwht_signs = signs_6144 ? (signs_6144 + w * 1024) : NULL;
        qurt_sem_up(&g_sem_start[w]);
    }
    for (int w = 0; w < NUM_WORKERS; w++) {
        qurt_sem_down(&g_sem_done[w]);
    }

    // Step 8: out_proj GEMV (5120 x 6144, prow = 1536)
    return hvx_gemv_q1(5120, 6144, 1536, no, out_bits, out_scales, y);
}

static float g_dsp_layer_h[5120]    __attribute__((aligned(128)));
static float g_dsp_layer_xn[5120]   __attribute__((aligned(128)));
static float g_dsp_layer_yo[5120]   __attribute__((aligned(128)));
static float g_dsp_lm_logits[124160] __attribute__((aligned(128)));

static void dsp_argmax_n(const float* buf, int n, int r_base, float* out_y) {
    const HVX_UVector* vBuf = (const HVX_UVector*)buf;
    int n_vec = n >> 5;
    HVX_Vector vBestVal = vBuf[0];
    HVX_Vector vCurIdx  = *(const HVX_Vector*)hvx_lane_idx;
    HVX_Vector vBestIdx = vCurIdx;
    HVX_Vector vStep32  = Q6_V_vsplat_R(32);
    for (int g = 1; g < n_vec; g++) {
        vCurIdx = Q6_Vw_vadd_VwVw(vCurIdx, vStep32);
        HVX_Vector vVal = vBuf[g];
        HVX_VectorPred qGt = Q6_Q_vcmp_gt_VsfVsf(vVal, vBestVal);
        vBestVal = Q6_V_vmux_QVV(qGt, vVal, vBestVal);
        vBestIdx = Q6_V_vmux_QVV(qGt, vCurIdx, vBestIdx);
    }
    float vals[32]   __attribute__((aligned(128)));
    int32_t idxs[32] __attribute__((aligned(128)));
    *(HVX_Vector*)vals = vBestVal;
    *(HVX_Vector*)idxs = vBestIdx;
    float best_v = vals[0];
    int32_t best_i = idxs[0];
    for (int k = 1; k < 32; k++) {
        if (vals[k] > best_v || (vals[k] == best_v && idxs[k] < best_i)) {
            best_v = vals[k];
            best_i = idxs[k];
        }
    }
    int32_t global_i = r_base + best_i;
    out_y[0] = best_v;
    memcpy(&out_y[1], &global_i, 4);
}

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
                        float* y) {
    init_thread_pool();

    if (layer_idx == -100 || layer_idx == -101) {
        if (layer_idx == -100) {
            memcpy(g_dsp_lin_x_fwht, x, 5120 * sizeof(float));
            if (signs_5120) {
                for (int w = 0; w < 5; w++) {
                    dsp_fwht1024_block(g_dsp_lin_x_fwht + w * 1024, signs_5120 + w * 1024);
                }
            }
        }
        int r0 = hvx_gemv_q1(31040, 5120, 1280, g_dsp_lin_x_fwht, in_bits,   in_scales,   g_dsp_lm_logits + 0 * 31040);
        if (r0 != 0) return r0;
        int r1 = hvx_gemv_q1(31040, 5120, 1280, g_dsp_lin_x_fwht, out_bits,  out_scales,  g_dsp_lm_logits + 1 * 31040);
        if (r1 != 0) return r1;
        int r2 = hvx_gemv_q1(31040, 5120, 1280, g_dsp_lin_x_fwht, gate_bits, gate_scales, g_dsp_lm_logits + 2 * 31040);
        if (r2 != 0) return r2;
        int r3 = hvx_gemv_q1(31040, 5120, 1280, g_dsp_lin_x_fwht, down_bits, down_scales, g_dsp_lm_logits + 3 * 31040);
        if (r3 != 0) return r3;
        dsp_argmax_n(g_dsp_lm_logits, 124160, (layer_idx == -100) ? 0 : 124160, y);
        return 0;
    }

    if (layer_idx <= -200 && layer_idx >= -203) {
        if (layer_idx == -200) {
            memcpy(g_dsp_lin_x_fwht, x, 5120 * sizeof(float));
            if (signs_5120) {
                for (int w = 0; w < 5; w++) {
                    dsp_fwht1024_block(g_dsp_lin_x_fwht + w * 1024, signs_5120 + w * 1024);
                }
            }
        }
        int r0 = hvx_gemv_q1(31040, 5120, 1280, g_dsp_lin_x_fwht, in_bits,  in_scales,  g_dsp_lm_logits + 0 * 31040);
        if (r0 != 0) return r0;
        int r1 = hvx_gemv_q1(31040, 5120, 1280, g_dsp_lin_x_fwht, out_bits, out_scales, g_dsp_lm_logits + 1 * 31040);
        if (r1 != 0) return r1;
        dsp_argmax_n(g_dsp_lm_logits, 62080, (-200 - layer_idx) * 62080, y);
        return 0;
    }

    const float* post_norm = NULL;
    int rc = 0;

    if (layer_idx <= -10000) {
        int code = -10000 - layer_idx;
        int fi   = code & 15;
        int pos  = code >> 4;
        if (pos < 0 || pos >= g_dsp_kv_ctx_max || !kvk || !kvv || !kvks || !kvvs) return -4;

        const uint8_t* f_aux  = lin_aux + (size_t)48 * 1188736 + (size_t)fi * 43008;
        const float* in_norm  = (const float*)(f_aux + 0);
        post_norm             = (const float*)(f_aux + 20480);
        const float* qnw      = (const float*)(f_aux + 40960);
        const float* knw      = (const float*)(f_aux + 41984);
        const float* rope_row = (const float*)(lin_aux + (size_t)48 * 1188736 + (size_t)16 * 43008 + (size_t)pos * 256);

        memcpy(g_dsp_layer_h, x, 5120 * sizeof(float));

        // 1. Input RMSNorm + FWHT-5120 + qkv_proj GEMV (14336 x 5120)
        dsp_rmsnorm_5120(g_dsp_layer_h, in_norm, g_dsp_lin_x_fwht);
        if (signs_5120) {
            for (int w = 0; w < 5; w++) {
                dsp_fwht1024_block(g_dsp_lin_x_fwht + w * 1024, signs_5120 + w * 1024);
            }
        }
        rc = hvx_gemv_q1(14336, 5120, 1280, g_dsp_lin_x_fwht, in_bits, in_scales, g_dsp_lin_qkv_z);
        if (rc != 0) return rc;

        // 2. K-head RMSNorm + NeoX RoPE + BF16-packed Q8_0 KV-cache write (4 KV heads)
        float* kk = g_dsp_lin_qkv_z + 12288;
        const float* vv = g_dsp_lin_qkv_z + 13312;
        const HVX_UVector* vKnw = (const HVX_UVector*)knw;
        const HVX_UVector* vRope = (const HVX_UVector*)rope_row;
        HVX_Vector vCos = vRope[0], vSin = vRope[1];

        size_t kv_stride = g_dsp_turbo4 ? 32 : 128;
        uint32_t* K_layer = kvk  + (size_t)fi * 4 * g_dsp_kv_ctx_max * kv_stride;
        uint32_t* V_layer = kvv  + (size_t)fi * 4 * g_dsp_kv_ctx_max * kv_stride;
        float* KS_layer   = kvks + (size_t)fi * 4 * g_dsp_kv_ctx_max;
        float* VS_layer   = kvvs + (size_t)fi * 4 * g_dsp_kv_ctx_max;

        for (int hk = 0; hk < 4; hk++) {
            float* r = kk + hk * 256;
            HVX_UVector* vK = (HVX_UVector*)r;
            HVX_Vector v0 = vK[0], v1 = vK[1], v2 = vK[2], v3 = vK[3];
            HVX_Vector v4 = vK[4], v5 = vK[5], v6 = vK[6], v7 = vK[7];
            HVX_Vector s01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, v0), Q6_Vqf32_vmpy_VsfVsf(v1, v1));
            HVX_Vector s23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v2, v2), Q6_Vqf32_vmpy_VsfVsf(v3, v3));
            HVX_Vector s45 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v4, v4), Q6_Vqf32_vmpy_VsfVsf(v5, v5));
            HVX_Vector s67 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v6, v6), Q6_Vqf32_vmpy_VsfVsf(v7, v7));
            HVX_Vector s03 = Q6_Vqf32_vadd_Vqf32Vqf32(s01, s23);
            HVX_Vector s47 = Q6_Vqf32_vadd_Vqf32Vqf32(s45, s67);
            float sk = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(s03, s47)));
            float inv = 1.0f / sqrtf(sk * (1.0f / 256.0f) + 1e-6f);
            int32_t inv_bits;
            memcpy(&inv_bits, &inv, 4);
            HVX_Vector vInv = Q6_V_vsplat_R(inv_bits);

            v0 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v0, vInv), vKnw[0]);
            v1 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v1, vInv), vKnw[1]);
            vK[2] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v2, vInv), vKnw[2]);
            vK[3] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v3, vInv), vKnw[3]);
            vK[4] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v4, vInv), vKnw[4]);
            vK[5] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v5, vInv), vKnw[5]);
            vK[6] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v6, vInv), vKnw[6]);
            vK[7] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v7, vInv), vKnw[7]);

            vK[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, vCos), Q6_Vqf32_vmpy_VsfVsf(v1, vSin)));
            vK[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, vSin), Q6_Vqf32_vmpy_VsfVsf(v1, vCos)));

            if (g_dsp_turbo4) {
                dsp_q4_enc_128b(r,            K_layer + ((size_t)hk * g_dsp_kv_ctx_max + pos) * 32, &KS_layer[(size_t)hk * g_dsp_kv_ctx_max + pos]);
                dsp_q4_enc_128b(vv + hk * 256, V_layer + ((size_t)hk * g_dsp_kv_ctx_max + pos) * 32, &VS_layer[(size_t)hk * g_dsp_kv_ctx_max + pos]);
            } else {
                dsp_q8_enc_bf16_256(r,            K_layer + ((size_t)hk * g_dsp_kv_ctx_max + pos) * 128, &KS_layer[(size_t)hk * g_dsp_kv_ctx_max + pos]);
                dsp_q8_enc_bf16_256(vv + hk * 256, V_layer + ((size_t)hk * g_dsp_kv_ctx_max + pos) * 128, &VS_layer[(size_t)hk * g_dsp_kv_ctx_max + pos]);
            }
        }

        // 3. Parallel 6-Thread QuRT HVX Q-Norm + RoPE + GQA + Sigmoid Gate + Fused FWHT-6144
        for (int w = 0; w < NUM_WORKERS; w++) {
            g_tasks[w].task_type  = TASK_FULL_GQA;
            g_tasks[w].h0         = w * 4;
            g_tasks[w].h1         = (w + 1) * 4;
            g_tasks[w].c0         = pos + 1;
            g_tasks[w].qkv        = g_dsp_lin_qkv_z;
            g_tasks[w].sc         = g_dsp_sc_buf[w];
            g_tasks[w].nw         = qnw;
            g_tasks[w].cw         = rope_row;
            g_tasks[w].wa         = K_layer;
            g_tasks[w].wb         = V_layer;
            g_tasks[w].alog       = KS_layer;
            g_tasks[w].dtb        = VS_layer;
            g_tasks[w].no         = g_dsp_lin_no;
            g_tasks[w].fwht_signs = signs_6144 ? (signs_6144 + w * 1024) : NULL;
            qurt_sem_up(&g_sem_start[w]);
        }
        for (int w = 0; w < NUM_WORKERS; w++) {
            qurt_sem_down(&g_sem_done[w]);
        }

        // 4. out_proj GEMV (5120 x 6144, prow = 1536)
        rc = hvx_gemv_q1(5120, 6144, 1536, g_dsp_lin_no, out_bits, out_scales, g_dsp_layer_yo);
        if (rc != 0) return rc;
    } else if (layer_idx < 0) {
        int fi = -1 - layer_idx;
        post_norm = (const float*)(lin_aux + (size_t)48 * 1188736 + (size_t)fi * 43008 + 20480);
        memcpy(g_dsp_layer_h, x, 5120 * sizeof(float));
        memcpy(g_dsp_lin_no, x + 5120, 6144 * sizeof(float));
        if (signs_6144) {
            for (int w = 0; w < 6; w++) {
                dsp_fwht1024_block(g_dsp_lin_no + w * 1024, signs_6144 + w * 1024);
            }
        }
        rc = hvx_gemv_q1(5120, 6144, 1536, g_dsp_lin_no, out_bits, out_scales, g_dsp_layer_yo);
        if (rc != 0) return rc;
    } else {
        const uint8_t* aux_layer = lin_aux + (size_t)layer_idx * 1188736;
        const float* in_norm = (const float*)(aux_layer + 1147776);
        post_norm            = (const float*)(aux_layer + 1168256);

        memcpy(g_dsp_layer_h, x, 5120 * sizeof(float));

        // 1. Input RMSNorm
        dsp_rmsnorm_5120(g_dsp_layer_h, in_norm, g_dsp_layer_xn);

        // 2. Fused Linear Attention Block
        rc = hvx_lin_attn_fused(layer_idx, g_dsp_layer_xn,
                                in_bits, in_scales,
                                out_bits, out_scales,
                                ssm_conv, ssm_rec, lin_aux,
                                signs_5120, signs_6144,
                                g_dsp_layer_yo);
        if (rc != 0) return rc;
    }

    // 3. Residual 1 (h += yo)
    HVX_UVector* vh = (HVX_UVector*)g_dsp_layer_h;
    const HVX_UVector* vyo = (const HVX_UVector*)g_dsp_layer_yo;
    for (int g = 0; g < 160; g++) {
        vh[g] = Q6_Vsf_vadd_VsfVsf(vh[g], vyo[g]);
    }

    // 4. Post-Attention RMSNorm
    dsp_rmsnorm_5120(g_dsp_layer_h, post_norm, g_dsp_layer_xn);

    // 5. HVX MLP Input FWHT-5120 directly in L1 cache (zero semaphore barrier!)
    if (signs_5120) {
        for (int w = 0; w < 5; w++) {
            dsp_fwht1024_block(g_dsp_layer_xn + w * 1024, signs_5120 + w * 1024);
        }
    }

    // 6. Fused MLP: gate_up GEMV -> SwiGLU + FWHT-1024 -> down_proj GEMV
    rc = hvx_gemv_q1(34816, 5120, 1280, g_dsp_layer_xn, gate_bits, gate_scales, mlp_gate_up_buf);
    if (rc != 0) return rc;

    rc = hvx_swiglu_fwht1024(1, mlp_gate_up_buf, signs_17408, mlp_fwht_buf);
    if (rc != 0) return rc;

    rc = hvx_gemv_q1(5120, 17408, 4352, mlp_fwht_buf, down_bits, down_scales, g_dsp_layer_yo);
    if (rc != 0) return rc;

    // 7. Residual 2 (y = h + yo)
    HVX_UVector* vy = (HVX_UVector*)y;
    for (int g = 0; g < 160; g++) {
        vy[g] = Q6_Vsf_vadd_VsfVsf(vh[g], vyo[g]);
    }
    return 0;
}


